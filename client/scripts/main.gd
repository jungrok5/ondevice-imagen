extends Control
##
## Phase 1 main scene — verify the *workflow* end to end on Android:
##
##   button press
##     ↓
##   ContextProvider snapshot (date / time / weather / place / lat-lon)
##     ↓
##   DrawMomentPlugin.start_inference(prompt) — Kotlin Foreground Service
##     ↓
##   (~30 s of fake delay, no ONNX yet)
##     ↓
##   inference_completed signal + system notification
##     ↓
##   user taps notification → app foregrounds → result screen shows the PNG
##
## Phase 2 will replace the fake-delay block inside the Worker with the
## real ONNX Runtime SD 1.5 inference. Everything else (UI, signal wire,
## notifications) stays.
##

const ContextProvider := preload("res://scripts/context_provider.gd")

var ctx_provider: ContextProvider
var draw_plugin   # Engine singleton from android-plugin/, may be null on desktop.

# UI
var lat_edit: LineEdit
var lon_edit: LineEdit
var refresh_button: Button
var generate_button: Button
var status_label: Label
var ctx_label: Label
var detail_label: Label
var result_rect: TextureRect

var _last_output_path: String = ""


func _ready() -> void:
	ctx_provider = ContextProvider.new()
	add_child(ctx_provider)
	ctx_provider.context_changed.connect(_on_context_changed)

	if Engine.has_singleton("DrawMomentPlugin"):
		draw_plugin = Engine.get_singleton("DrawMomentPlugin")
		draw_plugin.connect("inference_completed", Callable(self, "_on_inference_completed"))
		draw_plugin.connect("inference_failed",    Callable(self, "_on_inference_failed"))
	else:
		draw_plugin = null

	_build_ui()
	_on_context_changed(ctx_provider.get_cached_context())
	# Initial network refresh — does nothing fatal on desktop without network.
	ctx_provider.set_location(ContextProvider.DEFAULT_LAT, ContextProvider.DEFAULT_LON)


func _build_ui() -> void:
	var root := VBoxContainer.new()
	root.anchor_right = 1.0
	root.anchor_bottom = 1.0
	root.add_theme_constant_override("separation", 8)
	root.set("offset_left", 16)
	root.set("offset_right", -16)
	root.set("offset_top", 16)
	root.set("offset_bottom", -16)
	add_child(root)

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
	refresh_button.text = "↻"
	refresh_button.tooltip_text = "Re-fetch weather + place from new lat/lon"
	refresh_button.custom_minimum_size = Vector2(56, 56)
	refresh_button.add_theme_font_size_override("font_size", 24)
	refresh_button.pressed.connect(_on_refresh_pressed)
	loc_row.add_child(refresh_button)

	# Generate
	generate_button = Button.new()
	generate_button.text = "Draw this moment"
	generate_button.custom_minimum_size = Vector2(0, 96)
	generate_button.add_theme_font_size_override("font_size", 28)
	generate_button.pressed.connect(_on_generate_pressed)
	root.add_child(generate_button)

	# Live context label
	ctx_label = Label.new()
	ctx_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	root.add_child(ctx_label)

	# Status (busy / done / error)
	status_label = Label.new()
	status_label.text = ("ready" if draw_plugin != null
		else "ready (no native plugin — desktop build, fake delay only)")
	root.add_child(status_label)

	# Detail (output path / error message)
	detail_label = Label.new()
	detail_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	detail_label.add_theme_font_size_override("font_size", 16)
	root.add_child(detail_label)

	# Result image
	result_rect = TextureRect.new()
	result_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	result_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	result_rect.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(result_rect)


func _row(parent: Node, label: String) -> HBoxContainer:
	var hbox := HBoxContainer.new()
	hbox.custom_minimum_size = Vector2(0, 56)
	var lbl := Label.new()
	lbl.text = label
	lbl.custom_minimum_size = Vector2(120, 0)
	hbox.add_child(lbl)
	parent.add_child(hbox)
	return hbox


func _on_refresh_pressed() -> void:
	var lat := float(lat_edit.text)
	var lon := float(lon_edit.text)
	ctx_provider.set_location(lat, lon)
	status_label.text = "fetching weather + place ..."


func _on_context_changed(ctx: Dictionary) -> void:
	ctx_label.text = "ctx: %s · %s · %s · %s · %s" % [
		String(ctx.get("date", "?")),
		String(ctx.get("time", "?")),
		String(ctx.get("weather", "?")),
		String(ctx.get("place_type", "?")),
		String(ctx.get("season", "?")),
	]


func _on_generate_pressed() -> void:
	var ctx := ctx_provider.get_cached_context()
	# A minimal prompt string the worker will pass to inference. Exact
	# format is finalised once the prompt builder is ported to the device
	# (or kept on PC and shipped as a precomputed string per cell).
	var prompt := "%s | %s | %s | %s | place=%s" % [
		String(ctx.get("date", "")),
		String(ctx.get("time", "")),
		String(ctx.get("weather", "")),
		String(ctx.get("season", "")),
		String(ctx.get("place_type", "")),
	]
	# Output path the Worker writes the finished PNG to. Both desktop
	# and Android should be able to write to user://.
	var out_path := "user://moment_%d.png" % int(Time.get_unix_time_from_system())
	_last_output_path = out_path

	generate_button.disabled = true
	status_label.text = "starting background inference..."
	detail_label.text = "prompt: %s\noutput: %s" % [prompt, out_path]

	if draw_plugin != null:
		draw_plugin.call("start_inference", prompt, ProjectSettings.globalize_path(out_path))
	else:
		# Desktop fallback — simulate the worker right here so the UI
		# wire-up is testable without an Android device.
		_desktop_fake_inference(prompt, out_path)


func _on_inference_completed(output_path: String) -> void:
	_load_result(output_path)
	generate_button.disabled = false


func _on_inference_failed(error_msg: String) -> void:
	status_label.text = "FAILED: " + error_msg
	generate_button.disabled = false


func _load_result(path: String) -> void:
	var img := Image.new()
	# Worker writes an absolute fs path; convert back to a Godot virtual
	# path for desktop runs.
	var godot_path := path
	if not path.begins_with("user://") and not path.begins_with("res://"):
		# `path` is already absolute; Image.load() accepts that.
		var err := img.load(path)
		if err != OK:
			status_label.text = "PNG load error: %d" % err
			return
	else:
		var err := img.load(ProjectSettings.globalize_path(godot_path))
		if err != OK:
			status_label.text = "PNG load error: %d" % err
			return
	result_rect.texture = ImageTexture.create_from_image(img)
	status_label.text = "done — " + path


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
