
import platform, asyncio, vk, weakref
from ctypes import cast, c_char_p, c_uint, c_ubyte, c_ulonglong, pointer, POINTER, byref, c_float, Structure, sizeof, memmove
from xmath import Mat4, perspective, translate, rotate
from os.path import dirname
from vk_base import *

class Vertex(Structure):
    _fields_ = (('pos', c_float*3), ('col', c_float*2))

class TextureApplication(Application):
    VERTEX_BUFFER_BIND_ID = 0
    def create_semaphores(self):
        create_info = vk.SemaphoreCreateInfo(
            s_type=vk.STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            next=None, flags=0
        )

        present = vk.Semaphore(0)
        render = vk.Semaphore(0)

        result1 = self.CreateSemaphore(self.device, byref(create_info), None, byref(present))
        result2 = self.CreateSemaphore(self.device, byref(create_info), None, byref(render))
        if vk.SUCCESS not in (result1, result2):
            raise RuntimeError('Failed to create the semaphores')

        self.render_semaphores['present'] = present
        self.render_semaphores['render'] = render
    def describe_bindings(self):
        bindings = (vk.VertexInputBindingDescription*1)()
        attributes = (vk.VertexInputAttributeDescription*2)()

        bindings[0].binding = self.VERTEX_BUFFER_BIND_ID
        bindings[0].stride = sizeof(Vertex)
        bindings[0].input_rate = vk.VERTEX_INPUT_RATE_VERTEX

        # Attribute descriptions
		# Describes memory layout and shader attribute locations

        # Location 0: Position
        attributes[0].binding = self.VERTEX_BUFFER_BIND_ID
        attributes[0].location = 0
        attributes[0].format = vk.FORMAT_R32G32B32_SFLOAT
        attributes[0].offset = 0

        # Location 1: Color
        attributes[1].binding = self.VERTEX_BUFFER_BIND_ID
        attributes[1].location = 1
        attributes[1].format = vk.FORMAT_R32G32_SFLOAT
        attributes[1].offset = sizeof(c_float)*3

        self.triangle['bindings'] = bindings
        self.triangle['attributes'] = attributes
    def create


def main():
    app = TextureApplication()
    app.run()

    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == '__main__':
    main()