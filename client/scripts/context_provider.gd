extends Node
##
## Environmental context — same pattern as WaterBloom's context_provider.gd.
##
## Pulls (lat, lon) → (weather, place_type, season, time_of_day) and
## caches the result in memory. The image picker reads it via
## get_cached_context() — synchronous, never blocks on network. If the
## fetch hasn't completed yet, callers see the offline fallback (Seoul
## defaults + system-clock-derived season/time).
##
## Phase 1 (this file): Open-Meteo for weather, Nominatim for place.
## GPS comes from a desktop-friendly lat/lon LineEdit until phase 2
## wires the Android plugin.
##

const REFRESH_INTERVAL_SEC := 30 * 60

# Default fallback coords. Phase 2 replaces with actual GPS on Android.
const DEFAULT_LAT := 37.5665   # Seoul City Hall
const DEFAULT_LON := 126.9780

const _OPEN_METEO := "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=weather_code"
const _NOMINATIM  := "https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=14"
# Nominatim usage policy requires a User-Agent identifying the app +
# a contact URL. Repo URL is the stable contact while the app name and
# domain are TBD.
const _UA_HEADER := "User-Agent: local-ai-rnd/0.1 (+https://github.com/jungrok5/localairnd)"

signal context_changed(ctx: Dictionary)

var _cached: Dictionary = {}
var _last_refresh_unix: int = 0
var _http_weather: HTTPRequest
var _http_geo: HTTPRequest


func _ready() -> void:
	_http_weather = HTTPRequest.new()
	add_child(_http_weather)
	_http_weather.request_completed.connect(_on_weather_response)

	_http_geo = HTTPRequest.new()
	add_child(_http_geo)
	_http_geo.request_completed.connect(_on_geocode_response)

	_cached = _build_offline_context(DEFAULT_LAT, DEFAULT_LON)


# Called from main.gd when the user changes lat/lon (or, in phase 2,
# the Android GPS plugin emits new coordinates).
func set_location(lat: float, lon: float) -> void:
	_cached.lat = lat
	_cached.lon = lon
	# Re-fetch immediately. Don't wait for the 30-min refresh cadence
	# when the user has explicitly moved.
	_refresh()


func _process(_delta: float) -> void:
	var now := int(Time.get_unix_time_from_system())
	if now - _last_refresh_unix > REFRESH_INTERVAL_SEC:
		_refresh()


func get_cached_context() -> Dictionary:
	# Re-stamp time/season at read time so a cached daytime context
	# doesn't keep its old "day" value once it's evening.
	var ctx := _cached.duplicate()
	ctx.season = _season_now()
	ctx.time_of_day = _time_of_day_now()
	ctx.date = _date_string_now()
	ctx.time = _time_string_now()
	return ctx


func _refresh() -> void:
	_last_refresh_unix = int(Time.get_unix_time_from_system())
	var lat: float = float(_cached.get("lat", DEFAULT_LAT))
	var lon: float = float(_cached.get("lon", DEFAULT_LON))

	# IMPORTANT: do NOT wipe `_cached` here — the HTTP requests are async
	# and `get_cached_context()` is still serving reads in the meantime.
	# The response handlers update individual fields on success; on
	# failure they log and leave the old values in place.
	var weather_url: String = _OPEN_METEO.format({ "lat": lat, "lon": lon })
	var w_err := _http_weather.request(weather_url)
	if w_err != OK:
		print("[Context] weather request init failed: %d" % w_err)

	var geo_url: String = _NOMINATIM.format({ "lat": lat, "lon": lon })
	var g_err := _http_geo.request(geo_url, [_UA_HEADER])
	if g_err != OK:
		print("[Context] geocode request init failed: %d" % g_err)


func _on_weather_response(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		print("[Context] weather fetch failed result=%d code=%d (offline fallback)" % [result, code])
		return
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) != TYPE_DICTIONARY:
		return
	var current = parsed.get("current", {})
	var w_code := int(current.get("weather_code", -1))
	_cached.weather = _wmo_code_to_token(w_code)
	print("[Context] weather updated: %s (WMO %d)" % [String(_cached.weather), w_code])
	context_changed.emit(get_cached_context())


func _on_geocode_response(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if result != HTTPRequest.RESULT_SUCCESS or code != 200:
		print("[Context] geocode fetch failed result=%d code=%d" % [result, code])
		return
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if typeof(parsed) != TYPE_DICTIONARY:
		return
	var address: Dictionary = parsed.get("address", {})
	var country_code := String(address.get("country_code", "")).to_upper()
	if country_code != "":
		_cached.country = country_code
	_cached.place_type = _detect_place_type(address)
	# Display name (Korean if available) for UI.
	_cached.display_name = String(parsed.get("display_name", _cached.get("display_name", "")))
	# Preserve the raw address dict so the prompt builder can mine
	# road / suburb / city / building / amenity / leisure / tourism / etc.
	_cached.address = address
	# Build a short, human-readable POI summary (English where possible)
	# that the prompt builder can splice straight into the SD prompt.
	_cached.poi_summary = _build_poi_summary(address, _cached.display_name)
	print("[Context] geo updated: country=%s place_type=%s poi=%s" % [
		String(_cached.country), String(_cached.place_type), String(_cached.poi_summary)
	])
	context_changed.emit(get_cached_context())


func _build_poi_summary(address: Dictionary, display_name: String) -> String:
	# A few-word location label that captures what kind of place we're in,
	# preferring the most specific tag available. SD prompts react well to
	# concrete proper nouns ("Seoul City Hall") and concrete amenity names
	# ("a cafe", "a riverside park") — so we hand both when we have them.
	var parts: Array = []
	var named: String = ""
	for key in ["amenity", "leisure", "tourism", "historic", "shop",
				"office", "building", "natural", "man_made", "public_building"]:
		if address.has(key) and String(address[key]).length() > 0:
			named = String(address[key]).replace("_", " ")
			parts.append("a " + named)
			break
	# Most specific area name (suburb / neighbourhood / city / town).
	for key in ["suburb", "neighbourhood", "quarter", "city_district",
				"city", "town", "village", "county"]:
		if address.has(key) and String(address[key]).length() > 0:
			parts.append(String(address[key]))
			break
	if address.has("road") and String(address.road).length() > 0:
		parts.append("near " + String(address.road))
	if parts.is_empty() and display_name.length() > 0:
		# Fallback: first 2 components of the comma-separated display_name.
		var dn_parts: PackedStringArray = display_name.split(",")
		if dn_parts.size() > 0:
			parts.append(String(dn_parts[0]).strip_edges())
		if dn_parts.size() > 1:
			parts.append(String(dn_parts[1]).strip_edges())
	return ", ".join(parts)


func _detect_place_type(address: Dictionary) -> String:
	# Map OSM tags to the 5 catalog event tags. Pattern adapted from
	# WaterBloom's context_provider but the bucket vocabulary matches
	# our scripts/lora_catalog.py event keys (e1..e5).
	if address.has("amenity"):
		var a := String(address.amenity).to_lower()
		if a in ["cafe", "coffee_shop", "restaurant", "fast_food", "food_court"]:
			return "cafe"        # → e2
		if a in ["library"]:
			return "library"     # → e5
		if a in ["marketplace"]:
			return "market"      # → e4
	if address.has("leisure"):
		var l := String(address.leisure).to_lower()
		if l in ["park", "garden", "playground"]:
			return "river"       # → e3 (river/park outdoor night)
	if address.has("natural"):
		var n := String(address.natural).to_lower()
		if n in ["water", "bay"]:
			return "river"       # → e3
	if address.has("shop"):
		var s := String(address.shop).to_lower()
		if s in ["mall", "department_store"]:
			return "market"      # → e4
	# residential / nondescript address → indoor home
	return "home"                # → e1


func _build_offline_context(lat: float, lon: float) -> Dictionary:
	return {
		"season":       _season_now(),
		"time_of_day":  _time_of_day_now(),
		"weather":      "unknown",
		"country":      "KR",
		"place_type":   "home",
		"display_name": "",
		"address":      {},
		"poi_summary":  "",
		"lat":           lat,
		"lon":           lon,
	}


func _season_now() -> String:
	var m := int(Time.get_datetime_dict_from_system().month)
	if m == 12 or m == 1 or m == 2:
		return "winter"
	if m >= 3 and m <= 5:
		return "spring"
	if m >= 6 and m <= 8:
		return "summer"
	return "autumn"


func _time_of_day_now() -> String:
	var h := int(Time.get_datetime_dict_from_system().hour)
	if h >= 22 or h < 5:
		return "night"
	if h >= 17:
		return "evening"
	if h >= 11:
		return "noon"
	return "morning"


func _date_string_now() -> String:
	var d := Time.get_datetime_dict_from_system()
	return "%04d-%02d-%02d" % [int(d.year), int(d.month), int(d.day)]


func _time_string_now() -> String:
	var d := Time.get_datetime_dict_from_system()
	return "%02d:%02d" % [int(d.hour), int(d.minute)]


func _wmo_code_to_token(code: int) -> String:
	# WMO weather code per Open-Meteo. https://open-meteo.com/en/docs
	# Bucket mapping aligned with src/event_prompt.py _WEATHER_LOOKUP keys.
	match code:
		0:
			return "clear"     # 맑음
		1, 2, 3:
			return "overcast"  # 흐림 / 구름
		45, 48:
			return "fog"
		51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82:
			return "rain"      # 비
		71, 73, 75, 77, 85, 86:
			return "snow"      # 눈
		95, 96, 99:
			return "rain"      # thunder → treat as heavy rain for matching
		_:
			return "unknown"
