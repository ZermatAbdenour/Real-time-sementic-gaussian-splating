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

        self._last_main_w = 0
        self._last_main_h = 0


        self.is_open= True
        self.is_hovered = False
        self.is_focused = False

        self.window_pos = (0, 0)
        self.window_size = (800, 600)


    def draw(self,delta_time):
        #render and than draw the window viewport
        self.renderer.Render()

        imgui.begin(self.title)

        self.is_hovered = imgui.is_window_hovered()
        self.is_focused = imgui.is_window_focused()
        self.window_pos = imgui.get_window_pos()
        self.window_size = imgui.get_window_size()


        avail_w, avail_h = imgui.get_content_region_avail()
        target_w = max(self.min_w, int(avail_w))
        target_h = max(self.min_h, int(avail_h))

        if target_w != int(self.fb.width) or target_h != int(self.fb.height):
            self.fb.resize(target_w, target_h)
            self.renderer.on_resize()

        # Flip UVs vertically if needed
        uv0, uv1 = ((0.0, 1.0), (1.0, 0.0)) if self.flip_y else ((0.0, 0.0), (1.0, 1.0))

        tex_ref = imgui.ImTextureRef(int(self.fb.color_tex))
        image_size = imgui.ImVec2(float(self.fb.width), float(self.fb.height))
        uv0_v = imgui.ImVec2(float(uv0[0]), float(uv0[1]))
        uv1_v = imgui.ImVec2(float(uv1[0]), float(uv1[1]))

        imgui.image(tex_ref, image_size, uv0=uv0_v, uv1=uv1_v)

        imgui.end()
        self.process_input(context.window,delta_time)

        return int(self.fb.width), int(self.fb.height)
    
    def process_input(self, window, delta_time):
        if not self.is_focused and not self.is_hovered:
            return
        
        io = imgui.get_io()
        
        # Scroll
        if self.is_hovered and io.mouse_wheel != 0:
            self.renderer.camera.process_scroll(io.mouse_wheel)
        
        # Mouse look (only when right mouse button is held)
        if self.is_hovered and imgui.is_mouse_down(1):  # Right mouse button
            mouse_x, mouse_y = imgui.get_mouse_pos()
            self.renderer.camera.process_mouse(mouse_x, mouse_y, delta_time)
            #imgui.set_mouse_cursor(imgui.MOUSE_CURSOR_NONE)
        else:
            self.renderer.camera.first_mouse = True
            #imgui.set_mouse_cursor(imgui.MOUSE_CURSOR_ARROW)
        
        # Keyboard (only when focused)
        if self.is_focused:
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