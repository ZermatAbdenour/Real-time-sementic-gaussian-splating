from imgui_bundle import imgui
import glfw
from RTSGS.GUI import context

class ViewportWindow:
    def __init__(
        self,
        renderer,
        title: str = "Viewport",
        flip_y: bool = True,
        min_size=(1, 1),
    ):
        self.renderer = renderer
        self.fb = renderer.fb
        self.title = title
        self.flip_y = flip_y
        self.min_w, self.min_h = min_size

        self.is_open = True
        self.is_hovered = False
        self.is_focused = False
        
        # Track if we are currently rotating/looking
        self.is_looking = False

        self.window_pos = (0, 0)
        self.window_size = (800, 600)

    def draw(self, delta_time):
        # 1. Render the scene to the Framebuffer
        self.renderer.Render()

        # 2. Start ImGui Window
        imgui.begin(self.title)

        self.is_hovered = imgui.is_window_hovered()
        self.is_focused = imgui.is_window_focused()
        self.window_pos = imgui.get_window_pos()
        self.window_size = imgui.get_window_size()

        # Handle Resizing
        avail_w, avail_h = imgui.get_content_region_avail()
        target_w = max(self.min_w, int(avail_w))
        target_h = max(self.min_h, int(avail_h))

        if target_w != int(self.fb.width) or target_h != int(self.fb.height):
            self.fb.resize(target_w, target_h)
            self.renderer.on_resize()

        # 3. Draw the Texture
        uv0, uv1 = ((0.0, 1.0), (1.0, 0.0)) if self.flip_y else ((0.0, 0.0), (1.0, 1.0))
        tex_ref = imgui.ImTextureRef(int(self.fb.color_tex))
        image_size = imgui.ImVec2(float(self.fb.width), float(self.fb.height))
        
        imgui.image(
            tex_ref, 
            image_size, 
            uv0=imgui.ImVec2(float(uv0[0]), float(uv0[1])), 
            uv1=imgui.ImVec2(float(uv1[0]), float(uv1[1]))
        )

        # 4. Process Input (Using the context.window you imported)
        self.process_input(context.window, delta_time)

        imgui.end()
        
        return int(self.fb.width), int(self.fb.height)

    def process_input(self, window, delta_time):
        io = imgui.get_io()

        # --- MOUSE LOOK LOGIC (CAPTURE) ---
        # Start looking if Right Mouse (1) is clicked while hovered
        if imgui.is_mouse_clicked(1) and self.is_hovered:
            self.is_looking = True
            # Optional: Lock cursor to window for infinite rotation
            # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        # Stop looking when button is released
        if not imgui.is_mouse_down(1):
            if self.is_looking:
                self.is_looking = False
                # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self.renderer.camera.first_mouse = True

        # Update camera if we are currently in "looking" mode
        if self.is_looking:
            mouse_x, mouse_y = imgui.get_mouse_pos()
            self.renderer.camera.process_mouse(mouse_x, mouse_y, delta_time)

        # --- SCROLL ---
        if self.is_hovered and io.mouse_wheel != 0:
            self.renderer.camera.process_scroll(io.mouse_wheel)

        # --- KEYBOARD ---
        # Allow movement if focused OR if we are currently rotating the camera
        if self.is_focused or self.is_looking:
            keys = {
                'W': glfw.get_key(window, glfw.KEY_W) == glfw.PRESS,
                'A': glfw.get_key(window, glfw.KEY_A) == glfw.PRESS,
                'S': glfw.get_key(window, glfw.KEY_S) == glfw.PRESS,
                'D': glfw.get_key(window, glfw.KEY_D) == glfw.PRESS,
                'Q': glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS,
                'E': glfw.get_key(window, glfw.KEY_E) == glfw.PRESS,
                'SHIFT': glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS,
            }
            self.renderer.camera.process_keyboard(keys, delta_time)