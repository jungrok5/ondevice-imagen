extends Control
##
## Phase 1 main scene — verify the *workflow* end to end on Android,
## and also serve as a hand-held testbed:
##
##   [Build prompt]   ← bundles ctx + selected LoRA into an English prompt
##     ↓
##   prompt TextEdit  ← editable, the user can hand-tune before sending
##     ↓
##   [Generate]       ← starts the Foreground Service with the (possibly
##                       edited) prompt + LoRA trigger
##     ↓
##   inference_completed signal + system notification
##     ↓
##   result image + "elapsed: Xs" + "prompt sent: ..."
##
## Phase 2 will replace the fake-delay block inside the Worker with the
## real ONNX Runtime SD 1.5 inference. Everything else (UI, signal wire,
## notifications) stays.
##

const ContextProvider := preload("res://scripts/context_provider.gd")

# (display_label, lora_id, trigger_phrase) — trigger gets prepended to the
# user's prompt when "Build prompt" is pressed. lora_id is what Phase 2
# will pass to the native Worker so it picks the right .safetensors at
# inference time.
const LORA_PRESETS: Array = [
	["none — base SDXL",        "",                       ""],
	["worstimever (doodle)",    "worstimever_xl",         "WTE artstyle"],
	["MS Paint portrait",       "sdxl_mspaint_portraits", "MSPaint portrait"],
	["doodle-style (BTT)",      "doodle-style",           "SDXL_BTT_Doodle_v01"],
	["simple toons clean line", "simple-toons-style-sdxl",
		"a simple cartoon illustration, clean lineart"],
]

var ctx_provider: ContextProvider
var draw_plugin   # Engine singleton from android-plugin/, may be null on desktop.

# UI
var lat_edit: LineEdit
var lon_edit: LineEdit
var refresh_button: Button
var lora_picker: OptionButton
var build_button: Button
var prompt_edit: TextEdit
var generate_button: Button
var status_label: Label
var ctx_label: Label
var detail_label: Label
var result_rect: TextureRect

var _last_output_path: String = ""
var _started_at_msec: int = 0


func _ready() -> void:
	ctx_provider = ContextProvider.new()
	add_child(ctx_provider)
	ctx_provider.context_changed.connect(_on_context_changed)

	if Engine.has_singleton("DrawMomentPlugin"):
		draw_plugin = Engine.get_singleton("DrawMomentPlugin")
		draw_plugin.connect("inference_completed", Callable(self, "_on_inference_completed"))
		draw_plugin.connect("inference_failed",    Callable(self, "_on_inference_failed"))
		draw_plugin.connect("gps_updated",         Callable(self, "_on_gps_updated"))
		draw_plugin.connect("gps_failed",          Callable(self, "_on_gps_failed"))
	else:
		draw_plugin = null

	_build_ui()
	_on_context_changed(ctx_provider.get_cached_context())
	# Initial network refresh against the default fallback (Seoul City Hall).
	# A real GPS fix replaces it once the user grants the permission below.
	ctx_provider.set_location(ContextProvider.DEFAULT_LAT, ContextProvider.DEFAULT_LON)
	# On Android, ask for location permission up front. The plugin's
	# onMainRequestPermissionsResult fires request_current_location
	# automatically once the user grants — no race window. If the user
	# already granted on a previous launch, ask for a fix immediately.
	if draw_plugin != null:
		if draw_plugin.call("has_location_permission"):
			status_label.text = "asking GPS for a fix on launch..."
			draw_plugin.call("request_current_location")
		else:
			status_label.text = "requesting location permission..."
			draw_plugin.call("request_location_permission")


func _build_ui() -> void:
	# Wrap everything in a ScrollContainer so the result image at the
	# bottom doesn't squash the prompt editor and status labels above.
	var scroll := ScrollContainer.new()
	scroll.anchor_right = 1.0
	scroll.anchor_bottom = 1.0
	scroll.set("offset_left", 16)
	scroll.set("offset_right", -16)
	scroll.set("offset_top", 16)
	scroll.set("offset_bottom", -16)
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)

	var root := VBoxContainer.new()
	root.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.add_theme_constant_override("separation", 8)
	scroll.add_child(root)

	# Lat / Lon (phase 2 replaces with native GPS)
	var loc_row := _row(root, "Lat / Lon")
	lat_edit = LineEdit.new()
	lat_edit.text = str(ContextProvider.DEFAULT_LAT)
	lat_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	loc_row.add_child(lat_edit)
	lon_edit = LineEdit.new()
	lon_edit.text = str(ContextProvider.DEFAULT_LON)
	lon_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	loc_row.add_child(lon_edit)
	refresh_button = Button.new()
	refresh_button.text = "📍"
	refresh_button.tooltip_text = "Get my location (GPS) + re-fetch weather/place"
	refresh_button.custom_minimum_size = Vector2(64, 48)
	refresh_button.add_theme_font_size_override("font_size", 22)
	refresh_button.pressed.connect(_on_refresh_pressed)
	loc_row.add_child(refresh_button)

	# LoRA picker
	var lora_row := _row(root, "LoRA")
	lora_picker = OptionButton.new()
	lora_picker.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	for i in LORA_PRESETS.size():
		lora_picker.add_item(String(LORA_PRESETS[i][0]), i)
	lora_picker.select(1)  # default to worstimever — most-tested style
	lora_row.add_child(lora_picker)

	# Build prompt button
	build_button = Button.new()
	build_button.text = "Build prompt from context"
	build_button.custom_minimum_size = Vector2(0, 44)
	build_button.add_theme_font_size_override("font_size", 18)
	build_button.pressed.connect(_on_build_pressed)
	root.add_child(build_button)

	# Live context label (above the editable prompt — what we'll bundle)
	ctx_label = Label.new()
	ctx_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(ctx_label)

	# Editable prompt — user can hand-tune before pressing Generate.
	prompt_edit = TextEdit.new()
	prompt_edit.placeholder_text = "Press [Build prompt] to fill, then edit freely before [Generate]."
	prompt_edit.custom_minimum_size = Vector2(0, 140)
	prompt_edit.add_theme_font_size_override("font_size", 18)
	prompt_edit.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	root.add_child(prompt_edit)

	# Generate
	generate_button = Button.new()
	generate_button.text = "Generate"
	generate_button.custom_minimum_size = Vector2(0, 64)
	generate_button.add_theme_font_size_override("font_size", 24)
	generate_button.pressed.connect(_on_generate_pressed)
	root.add_child(generate_button)

	# Status (busy / done / error / elapsed) — autowrap so the long
	# done line ("done — elapsed 30.08s — /data/data/.../moment_*.png")
	# stays visible instead of getting truncated.
	status_label = Label.new()
	status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status_label.add_theme_font_size_override("font_size", 24)
	status_label.text = ("ready" if draw_plugin != null
		else "ready (no native plugin — desktop build, fake delay only)")
	root.add_child(status_label)

	# Detail (output path / sent prompt / error message)
	detail_label = Label.new()
	detail_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	detail_label.add_theme_font_size_override("font_size", 16)
	root.add_child(detail_label)

	# Result image — fixed minimum so it always shows but never expands
	# to crowd out the controls above. The ScrollContainer handles the
	# rest if the screen is short.
	result_rect = TextureRect.new()
	result_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	result_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	result_rect.custom_minimum_size = Vector2(0, 480)
	root.add_child(result_rect)


func _row(parent: Node, label: String) -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl := Label.new()
	lbl.text = label
	lbl.custom_minimum_size = Vector2(96, 0)
	hbox.add_child(lbl)
	parent.add_child(hbox)
	return hbox


func _on_refresh_pressed() -> void:
	# Two paths:
	#  - on Android with the plugin available, ask the OS for a fresh GPS
	#    fix; it lands in _on_gps_updated which then calls set_location.
	#  - on desktop or with no permission, fall back to whatever's already
	#    typed in the lat/lon fields.
	if draw_plugin != null:
		status_label.text = "asking GPS for a fix ..."
		draw_plugin.call("request_location_permission")
		draw_plugin.call("request_current_location")
		return
	var lat := float(lat_edit.text)
	var lon := float(lon_edit.text)
	ctx_provider.set_location(lat, lon)
	status_label.text = "fetching weather + place ..."


func _on_gps_updated(payload: String) -> void:
	# Plugin ships "lat,lon" as a single string to dodge Godot signal
	# numeric-arg plumbing.
	var parts: PackedStringArray = payload.split(",")
	if parts.size() != 2:
		status_label.text = "GPS bad payload: %s" % payload
		return
	var lat := float(parts[0])
	var lon := float(parts[1])
	# Echo into the editable fields so the user sees the actual coords
	# the rest of the pipeline is using.
	lat_edit.text = "%.5f" % lat
	lon_edit.text = "%.5f" % lon
	ctx_provider.set_location(lat, lon)
	status_label.text = "GPS fix → %.5f, %.5f — refreshing weather + place ..." % [lat, lon]


func _on_gps_failed(reason: String) -> void:
	# Don't block the rest of the UI — the default fallback context still
	# works, the user just gets the Seoul City Hall numbers.
	status_label.text = "GPS unavailable: %s — using default lat/lon" % reason


func _on_context_changed(ctx: Dictionary) -> void:
	var poi := String(ctx.get("poi_summary", ""))
	if poi.is_empty():
		poi = String(ctx.get("place_type", "?"))
	ctx_label.text = "ctx: %s · %s · %s · %s · %s\nPOI: %s" % [
		String(ctx.get("date", "?")),
		String(ctx.get("time", "?")),
		String(ctx.get("weather", "?")),
		String(ctx.get("season", "?")),
		String(ctx.get("time_of_day", "?")),
		poi,
	]


# --- Build prompt ------------------------------------------------------------

func _on_build_pressed() -> void:
	var ctx := ctx_provider.get_cached_context()
	prompt_edit.text = _build_full_prompt(ctx, _selected_lora_trigger())


func _selected_lora_trigger() -> String:
	var idx := lora_picker.get_selected_id()
	if idx < 0 or idx >= LORA_PRESETS.size():
		return ""
	return String(LORA_PRESETS[idx][2])


func _selected_lora_id() -> String:
	var idx := lora_picker.get_selected_id()
	if idx < 0 or idx >= LORA_PRESETS.size():
		return ""
	return String(LORA_PRESETS[idx][1])


func _build_full_prompt(ctx: Dictionary, lora_trigger: String) -> String:
	# Compose a SD-friendly English prompt from everything ContextProvider
	# managed to grab. Order matters for SD 1.5: trigger first, then the
	# scene description, then mood / time / weather. POI noun phrase
	# (e.g. "a cafe, Junggu, near Sejong-daero") goes near the front so
	# the model anchors its composition to it.
	var parts: Array = []
	if lora_trigger.length() > 0:
		parts.append(lora_trigger)

	var poi := String(ctx.get("poi_summary", ""))
	var place_type := String(ctx.get("place_type", ""))
	var weather := String(ctx.get("weather", ""))
	var season := String(ctx.get("season", ""))
	var time_of_day := String(ctx.get("time_of_day", ""))
	var date := String(ctx.get("date", ""))
	var time_str := String(ctx.get("time", ""))

	# Anchor sentence — concrete location if we have one, otherwise the
	# 5-bucket place_type (cafe / library / market / river / home).
	if poi.length() > 0:
		parts.append("a scene at %s" % poi)
	elif place_type.length() > 0:
		parts.append("a scene at a %s" % place_type)

	# Mood line — time of day + weather + season.
	var mood: Array = []
	if time_of_day.length() > 0:
		mood.append(time_of_day)
	if weather.length() > 0 and weather != "unknown":
		mood.append(weather + " weather")
	if season.length() > 0:
		mood.append("during " + season)
	if not mood.is_empty():
		parts.append(", ".join(mood))

	# Timestamp footnote — useful for "different result every day even at
	# the same place" experiments. SD 1.5 mostly ignores numeric strings,
	# but we keep them for the user's records (and to quietly perturb the
	# prompt embedding so two same-context generations diverge).
	if date.length() > 0 or time_str.length() > 0:
		parts.append("(%s %s)" % [date, time_str])

	return ", ".join(parts)


# --- Generate ---------------------------------------------------------------

func _on_generate_pressed() -> void:
	var prompt := prompt_edit.text.strip_edges()
	if prompt.is_empty():
		# Convenience: pressing Generate without ever pressing Build is
		# the common case, so build it on the fly.
		var ctx := ctx_provider.get_cached_context()
		prompt = _build_full_prompt(ctx, _selected_lora_trigger())
		prompt_edit.text = prompt

	# Output path the Worker writes the finished PNG to.
	var out_path := "user://moment_%d.png" % int(Time.get_unix_time_from_system())
	_last_output_path = out_path
	_started_at_msec = Time.get_ticks_msec()

	generate_button.disabled = true
	build_button.disabled = true
	status_label.text = "generating ..."
	detail_label.text = "lora: %s\nprompt sent:\n%s\noutput: %s" % [
		_selected_lora_id() if _selected_lora_id().length() > 0 else "(none)",
		prompt,
		out_path,
	]

	if draw_plugin != null:
		draw_plugin.call("start_inference", prompt, ProjectSettings.globalize_path(out_path))
	else:
		# Desktop fallback — simulate the worker right here so the UI
		# wire-up is testable without an Android device.
		_desktop_fake_inference(prompt, out_path)


func _on_inference_completed(output_path: String) -> void:
	var elapsed_ms := Time.get_ticks_msec() - _started_at_msec
	_load_result(output_path, elapsed_ms)
	generate_button.disabled = false
	build_button.disabled = false


func _on_inference_failed(error_msg: String) -> void:
	status_label.text = "FAILED: " + error_msg
	generate_button.disabled = false
	build_button.disabled = false


func _load_result(path: String, elapsed_ms: int) -> void:
	var img := Image.new()
	# Worker writes an absolute fs path; convert back to a Godot virtual
	# path for desktop runs.
	var godot_path := path
	var err: int
	if not path.begins_with("user://") and not path.begins_with("res://"):
		err = img.load(path)
	else:
		err = img.load(ProjectSettings.globalize_path(godot_path))
	if err != OK:
		status_label.text = "PNG load error %d at %s" % [err, path]
		return
	result_rect.texture = ImageTexture.create_from_image(img)
	status_label.text = "done — elapsed %.2fs — %s" % [elapsed_ms / 1000.0, path]


# --- Desktop fallback ---------------------------------------------------------

func _desktop_fake_inference(prompt: String, out_path: String) -> void:
	# Simulate ~3 s of work and write a tiny placeholder PNG so the UI
	# displays something. Phase 2 replaces this entire block with the
	# native Worker doing real ONNX inference.
	var t := Timer.new()
	t.wait_time = 3.0
	t.one_shot = true
	add_child(t)
	t.timeout.connect(func() -> void:
		var img := Image.create(512, 512, false, Image.FORMAT_RGB8)
		img.fill(Color(0.95, 0.92, 0.88))
		img.save_png(ProjectSettings.globalize_path(out_path))
		_on_inference_completed(out_path)
		t.queue_free()
	)
	t.start()
	# Touch `prompt` to silence "unused arg" warnings — it's logged in the
	# detail label already.
	prompt = prompt
