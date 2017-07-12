
import platform, asyncio, vk, weakref
from ctypes import cast, c_char_p, c_uint, c_ubyte, c_ulonglong, pointer, POINTER, byref,c_void_p, c_float,c_byte,c_ubyte,Structure, sizeof, memmove
from xmath import Mat4, perspective, translate, rotate
from os.path import dirname
from vk_base import *

class Vertex(Structure):
    _fields_ = (('pos', c_float*3), ('col', c_float*2) )

class Pixel(Structure):
    _fields_ = (('r', c_ubyte),('g', c_ubyte), ('b', c_ubyte), ('a', c_ubyte))

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

    def create_quad(self):
        data = c_void_p(0)

        # Setup vertices
        vertices_data = (Vertex*4)(
            Vertex(pos=(-1.0, -1.0, 0.0), col=(0.0, 0.0)),
            Vertex(pos=(1.0, -1.0, 0.0), col=(1.0, 0.0)),
            Vertex(pos=(1.0, 1.0, 0.0), col=(1.0, 1.0)),
            Vertex(pos=(-1.0, 1.0, 0.0), col=(0.0, 1.0))
        )

        vertices_size = sizeof(Vertex)*4

        # Setup indices
        indices_data = (c_uint*6)(0,1,3,1,2,3)
        indices_size = sizeof(indices_data)

        self.uploadToStaging(vertices_data, vertices_size)
        (buffer, bufmem) = self.createBuffer(
            vertices_size,
            vk.BUFFER_USAGE_VERTEX_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.copyBuffer(self.stagingBuffer, buffer, 0, vertices_size)

        self.uploadToStaging(indices_data, indices_size)
        (indiceBuffer, indiceMem) = self.createBuffer(
            indices_size,
            vk.BUFFER_USAGE_INDEX_BUFFER_BIT | vk.BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.copyBuffer(self.stagingBuffer, indiceBuffer, 0, indices_size)

        self.vertexBuffer = buffer
        self.vertexBufferMem = bufmem
        self.indexBuffer = indiceBuffer
        self.indexMem = indiceMem
        self.describe_bindings()


    def create_uniform_buffers(self):
        (self.uniformBuffer, self.uniformBufferMem) = self.createBuffer(
            sizeof(Mat4)*3,
            vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT)

        self.update_uniform_buffers()


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

        # Location 1: Texture Coordiate
        attributes[1].binding = self.VERTEX_BUFFER_BIND_ID
        attributes[1].location = 1
        attributes[1].format = vk.FORMAT_R32G32_SFLOAT
        attributes[1].offset = sizeof(c_float)*3

        self.bindings = bindings
        self.attributes = attributes

    def createImageTexture(self):
        print("create image Texture")
        texWidth = 32
        texHeight = 32
        texSize = texWidth * texHeight
        image_data = (Pixel * texSize)()
        imsize = sizeof(image_data)
        red = Pixel(255, 0, 0, 255)
        blue = Pixel(0, 0, 255, 255)
        green = Pixel(0, 255, 0, 255)
        gray = Pixel(128, 128, 128, 255)
        i = 0
        j = 0
        while i < texHeight:
            while j < texWidth:
                if i < 16:
                    if j < 16:
                        image_data[i * texWidth + j] = red
                    else:
                        image_data[i * texWidth + j] = blue
                else:
                    if j < 16:
                        image_data[i * texWidth + j] = green
                    else:
                        image_data[i * texWidth + j] = gray
                j += 1
            j = 0
            i += 1
        print("end memory mapping")
        self.uploadToStaging(image_data, imsize)
        (deviceImg, deviceImgMem) = self.create2dImage(
            texWidth, texHeight, vk.FORMAT_R8G8B8A8_UNORM,
            vk.IMAGE_TILING_OPTIMAL,
            vk.IMAGE_USAGE_TRANSFER_DST_BIT|vk.IMAGE_USAGE_SAMPLED_BIT,
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.deviceImg = deviceImg
        self.deviceImgMem = deviceImgMem

        self.copyBufferToImage(self.stagingBuffer, deviceImg, texWidth, texHeight)

    def update_uniform_buffers(self):
        data = c_void_p(0)
        matsize = sizeof(Mat4)*3
        self.zoom = -1.0               # Scene zoom
        self.rotation = (c_float*3)()  # Scene rotation
        # Projection
        width, height = self.window.dimensions()
        self.matrices[0].set_data(perspective(60.0, width/height, 0.1, 256.0))

        # Model
        mod_mat = rotate(None, self.rotation[0], (1.0, 0.0, 0.0))
        mod_mat = rotate(mod_mat, self.rotation[1], (0.0, 1.0, 0.0))
        self.matrices[1].set_data(rotate(mod_mat, self.rotation[2], (0.0, 0.0, 1.0)))

        # View
        self.matrices[2].set_data(translate(None, (0.0, 0.0, self.zoom)))


        self.MapMemory(self.device, self.uniformBufferMem, 0, matsize, 0, byref(data))
        memmove(data, self.matrices, matsize)
        self.UnmapMemory(self.device, self.uniformBufferMem)

    def create_descriptor_set_layout(self):
        # Setup layout of descriptors used in this example
		# Basically connects the different shader stages to descriptors
		# for binding uniform buffers, image samplers, etc.
		# So every shader binding should map to one descriptor set layout
		# binding

        # Binding 0 : Uniform buffer (Vertex shader)
        binding = vk.DescriptorSetLayoutBinding(
            binding=0,
            descriptor_type=vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptor_count=1, stage_flags=vk.SHADER_STAGE_VERTEX_BIT,
            immutable_samplers=None
        )

        samplerLayout = vk.DescriptorSetLayoutBinding(
            binding=1,
            descriptor_type=vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptor_count=1,
            stage_flags=vk.SHADER_STAGE_FRAGMENT_BIT,
            immutable_samplers=None
        )

        bindings = (vk.DescriptorSetLayoutBinding*2)(binding, samplerLayout)
        layout = vk.DescriptorSetLayoutCreateInfo(
            s_type=vk.STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            next=None, flags=0, binding_count=2, bindings=bindings
        )

        ds_layout = vk.DescriptorSetLayout(0)
        result = self.CreateDescriptorSetLayout(self.device, byref(layout), None, byref(ds_layout))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not create descriptor set layout')

        # Create the pipeline layout that is used to generate the rendering pipelines that
		# are based on this descriptor set layout
		# In a more complex scenario you would have different pipeline layouts for different
		# descriptor set layouts that could be reused
        pipeline_info = vk.PipelineLayoutCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, next=None,
            flags=0, set_layout_count=1, set_layouts=pointer(ds_layout),
            push_constant_range_count=0
        )

        pipeline_layout = vk.PipelineLayout(0)
        result = self.CreatePipelineLayout(
            self.device, byref(pipeline_info), None, byref(pipeline_layout))


        self.pipeline_layout = pipeline_layout
        self.descriptor_set_layout = ds_layout

    def create_pipeline(self):
        # Vertex input state
        input_state = vk.PipelineVertexInputStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, next=None, flags=0,
            vertex_binding_description_count = 1,
            vertex_attribute_description_count = 2,
            vertex_binding_descriptions = cast(self.bindings, POINTER(vk.VertexInputBindingDescription)),
            vertex_attribute_descriptions = cast(self.attributes, POINTER(vk.VertexInputAttributeDescription))
        )
        self.input_state = input_state

        # Vertex input state
		# Describes the topoloy used with this pipeline
        input_assembly_state = vk.PipelineInputAssemblyStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, next=None,
            flags=0, primitive_restart_enable=0,
            topology=vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  #This pipeline renders vertex data as triangle lists
        )

        # Rasterization state
        raster_state = vk.PipelineRasterizationStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, next=None,
            flags=0,
            polygon_mode=vk.POLYGON_MODE_FILL,          # Solid polygon mode
            cull_mode= vk.CULL_MODE_NONE,               # No culling
            front_face=vk.FRONT_FACE_CLOCKWISE,
            depth_clamp_enable=0, rasterizer_discard_enable=0,
            depth_bias_enable=0, line_width=1.0
        )

        # Color blend state
        # Describes blend modes and color masks
        blend_state = vk.PipelineColorBlendAttachmentState(
            color_write_mask=0xF, blend_enable=0
        )
        color_blend_state = vk.PipelineColorBlendStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, next=None,
            flags=0, logic_op_enable=0, attachment_count=1, attachments=pointer(blend_state)
        )

        # Viewport state
        viewport_state = vk.PipelineViewportStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count=1, scissor_count=1
        )

        # Enable dynamic states
		# Describes the dynamic states to be used with this pipeline
		# Dynamic states can be set even after the pipeline has been created
		# So there is no need to create new pipelines just for changing
		# a viewport's dimensions or a scissor box
        dynamic_states = (c_uint*2)(vk.DYNAMIC_STATE_VIEWPORT, vk.DYNAMIC_STATE_SCISSOR)
        dynamic_state = vk.PipelineDynamicStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, next=None,
            flags=0,dynamic_state_count=2,
            dynamic_states=cast(dynamic_states, POINTER(c_uint))
        )

        # Depth and stencil state
		# Describes depth and stenctil test and compare ops
        # Basic depth compare setup with depth writes and depth test enabled
		# No stencil used
        op_state = vk.StencilOpState(
            fail_op=vk.STENCIL_OP_KEEP, pass_op=vk.STENCIL_OP_KEEP,
            compare_op=vk.COMPARE_OP_ALWAYS
        )
        depth_stencil_state = vk.PipelineDepthStencilStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, next=None,
            flags=0, depth_test_enable=1, depth_write_enable=1,
            depth_compare_op=vk.COMPARE_OP_LESS_OR_EQUAL,
            depth_bounds_test_enable=0, stencil_test_enable=0,
            front=op_state, back=op_state
        )

        # Multi sampling state
        # No multi sampling used in this example
        multisample_state = vk.PipelineMultisampleStateCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, next=None,
            flags=0, rasterization_samples=vk.SAMPLE_COUNT_1_BIT
        )

        # Load shaders
		# Shaders are loaded from the SPIR-V format, which can be generated from glsl
        shader_stages = (vk.PipelineShaderStageCreateInfo * 2)(
            self.load_shader('texture.vert.spv', vk.SHADER_STAGE_VERTEX_BIT),
            self.load_shader('font.frag.spv', vk.SHADER_STAGE_FRAGMENT_BIT)
        )

        create_info = vk.GraphicsPipelineCreateInfo(
            s_type=vk.STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, next=None,
            flags=0, stage_count=2,
            stages=cast(shader_stages, POINTER(vk.PipelineShaderStageCreateInfo)),
            vertex_input_state=pointer(input_state),
            input_assembly_state=pointer(input_assembly_state),
            tessellation_state=None,
            viewport_state=pointer(viewport_state),
            rasterization_state=pointer(raster_state),
            multisample_state=pointer(multisample_state),
            depth_stencil_state=pointer(depth_stencil_state),
            color_blend_state=pointer(color_blend_state),
            dynamic_state=pointer(dynamic_state),
            layout=self.pipeline_layout,
            render_pass=self.render_pass,
            subpass=0,
            basePipelineHandle=vk.Pipeline(0),
            basePipelineIndex=0
        )

        pipeline = vk.Pipeline(0)
        result = self.CreateGraphicsPipelines(self.device, self.pipeline_cache, 1, byref(create_info), None, byref(pipeline))
        if result != vk.SUCCESS:
             raise RuntimeError('Failed to create the graphics pipeline')

        self.pipeline = pipeline

    def create_descriptor_pool(self):

        # We need to tell the API the number of max. requested descriptors per type
        # This example only uses one descriptor type (uniform buffer) and only
		# requests one descriptor of this type
        type_counts = (vk.DescriptorPoolSize * 2)(
            vk.DescriptorPoolSize(
                type=vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptor_count=1),
            vk.DescriptorPoolSize(
                type=vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptor_count=1)
        )

        # Create the global descriptor pool
		# All descriptors used in this example are allocated from this pool
        pool_create_info = vk.DescriptorPoolCreateInfo(
            s_type=vk.STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, next=None,
            flags=0, pool_size_count=2, pool_sizes=type_counts,
            max_sets=1
        )

        pool = vk.DescriptorPool(0)
        result = self.CreateDescriptorPool(self.device, byref(pool_create_info), None, byref(pool))

        self.descriptor_pool = pool

    def create_descriptor_set(self):
        # Update descriptor sets determining the shader binding points
		# For every binding point used in a shader there needs to be one
		# descriptor set matching that binding point

        descriptor_alloc = vk.DescriptorSetAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, next=None,
            descriptor_pool=self.descriptor_pool, descriptor_set_count=1,
            set_layouts=pointer(self.descriptor_set_layout)
        )

        descriptor_set = vk.DescriptorSet(0)
        result = self.AllocateDescriptorSets(self.device, byref(descriptor_alloc), byref(descriptor_set))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not allocate descriptor set')

        bufferInfo = vk.DescriptorBufferInfo(
            buffer=self.uniformBuffer,
            offset=0,
            range=sizeof(self.matrices)
        )

        imageInfo = vk.DescriptorImageInfo(
            image_layout=vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image_view=self.imageView,
            sampler=self.sampler
        )

        #Binding 0 : Uniform buffer
        write_sets = (vk.WriteDescriptorSet*2)(
            vk.WriteDescriptorSet(
                s_type=vk.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, next=None,
                dst_set=descriptor_set, descriptor_count=1,
                descriptor_type=vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                buffer_info=pointer(bufferInfo),
                dst_binding=0 # Binds this uniform buffer to binding point 0
            ),
            vk.WriteDescriptorSet(
                s_type=vk.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, next=None,
                dst_set=descriptor_set, descriptor_count=1,
                descriptor_type=vk.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                image_info=pointer(imageInfo),
                dst_binding=1 # Binds this uniform buffer to binding point 0
                )
        )
        self.UpdateDescriptorSets(self.device, 2, write_sets, 0, None)
        self.descriptor_set = descriptor_set

    def init_command_buffers(self):

        begin_info = vk.CommandBufferBeginInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, next=None
        )

        clear_values = (vk.ClearValue*2)()
        clear_values[0].color = vk.ClearColorValue((c_float*4)(0.1, 0.1, 0.1, 1.0))
        clear_values[1].depth_stencil = vk.ClearDepthStencilValue(depth=1.0, stencil=0)

        width, height = self.window.dimensions()
        render_area = vk.Rect2D(
            offset=vk.Offset2D(x=0, y=0),
            extent=vk.Extent2D(width=width, height=height)
        )
        render_pass_begin = vk.RenderPassBeginInfo(
            s_type=vk.STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, next=None,
            render_pass=self.render_pass, render_area=render_area,
            clear_value_count=2,
            clear_values = cast(clear_values, POINTER(vk.ClearValue))
        )

        for index, cmdbuf in enumerate(self.post_present_buffers):
            assert(self.BeginCommandBuffer(cmdbuf, byref(begin_info)) == vk.SUCCESS)

            subres = vk.ImageSubresourceRange(
                aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, base_mip_level=0,
                level_count=1, base_array_layer=0, layer_count=1,
            )

            barrier = vk.ImageMemoryBarrier(
                s_type=vk.STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, next=None,
                src_access_mask=0,
                dst_access_mask=vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                old_layout=vk.IMAGE_LAYOUT_PRESENT_SRC_KHR,
                new_layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
                dst_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
                image=self.swapchain.images[index],
                subresource_range=subres
            )

            self.CmdPipelineBarrier(
				cmdbuf,
				vk.PIPELINE_STAGE_ALL_COMMANDS_BIT,
				vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, None,
				0, None,
				1, byref(barrier));

            assert(self.EndCommandBuffer(cmdbuf) == vk.SUCCESS)

        for index, cmdbuf in enumerate(self.draw_buffers):
            assert(self.BeginCommandBuffer(cmdbuf, byref(begin_info)) == vk.SUCCESS)

            render_pass_begin.framebuffer = self.framebuffers[index]
            self.CmdBeginRenderPass(cmdbuf, byref(render_pass_begin), vk.SUBPASS_CONTENTS_INLINE)

            # Update dynamic viewport state
            viewport = vk.Viewport(
                x=0.0, y=0.0, width=float(width), height=float(height),
                min_depth=0.0, max_depth=1.0
            )
            self.CmdSetViewport(cmdbuf, 0, 1, byref(viewport))

            # Update dynamic scissor state
            scissor = render_area
            self.CmdSetScissor(cmdbuf, 0, 1, byref(scissor))

            # Bind descriptor sets describing shader binding points
            self.CmdBindDescriptorSets(cmdbuf, vk.PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout, 0, 1, byref(self.descriptor_set), 0, None)

            # Bind the rendering pipeline (including the shaders)
            self.CmdBindPipeline(cmdbuf, vk.PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)

            # Bind triangle vertices
            offsets = c_ulonglong(0)
            self.CmdBindVertexBuffers(cmdbuf, self.VERTEX_BUFFER_BIND_ID, 1, byref(self.vertexBuffer), byref(offsets))

            # Bind triangle indices
            self.CmdBindIndexBuffer(cmdbuf, self.indexBuffer, 0, vk.INDEX_TYPE_UINT32)

            # Draw indexed triangle
            self.CmdDrawIndexed(cmdbuf, 6, 1, 0, 0, 1)

            self.CmdEndRenderPass(cmdbuf)

            # Add a present memory barrier to the end of the command buffer
			# This will transform the frame buffer color attachment to a
			# new layout for presenting it to the windowing system integration
            subres = vk.ImageSubresourceRange(
                aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, base_mip_level=0,
                level_count=1, base_array_layer=0, layer_count=1,
            )

            barrier = vk.ImageMemoryBarrier(
                s_type=vk.STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, next=None,
                src_access_mask=vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dst_access_mask=vk.ACCESS_MEMORY_READ_BIT,
                old_layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                new_layout=vk.IMAGE_LAYOUT_PRESENT_SRC_KHR,
                src_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
                dst_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
                image=self.swapchain.images[index],
                subresource_range=subres
            )

            self.CmdPipelineBarrier(
				cmdbuf,
				vk.PIPELINE_STAGE_ALL_COMMANDS_BIT,
				vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0,
				0, None,
				0, None,
				1, byref(barrier))


            assert(self.EndCommandBuffer(cmdbuf) == vk.SUCCESS)

    def resize_display(self, width, height):
        if not self.initialized:
            return

        Application.resize_display(self, width, height)

        self.init_command_buffers()
        self.QueueWaitIdle(self.queue)
        self.DeviceWaitIdle(self.device)

        self.update_uniform_buffers()

    def run(self):
        """
            Add the render phase to the asyncio loop
        """
        asyncio.ensure_future(self.render())
        self.initialized = True

    def draw(self):
        current_buffer = c_uint(0)

        #  Get next image in the swap chain (back/front buffer)
        result = self.AcquireNextImageKHR(
            self.device, self.swapchain.swapchain, c_ulonglong(-1),
            self.render_semaphores['present'], vk.Fence(0), byref(current_buffer)
        )
        if result != vk.SUCCESS:
            raise Exception("Could not aquire next image from swapchain")

        cb = current_buffer.value

        prebuf = vk.CommandBuffer(self.post_present_buffers[cb])
        submit_info = vk.SubmitInfo(
            s_type=vk.STRUCTURE_TYPE_SUBMIT_INFO,
            command_buffer_count=1,
            command_buffers=pointer(prebuf)
        )
        assert(self.QueueSubmit(self.queue, 1, byref(submit_info), vk.Fence(0)) == vk.SUCCESS)
        assert(self.QueueWaitIdle(self.queue) == vk.SUCCESS)

        # The submit information structure contains a list of
		# command buffers and semaphores to be submitted to a queue
		# If you want to submit multiple command buffers, pass an array
        stages = c_uint(vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        drawbuf = vk.CommandBuffer(self.draw_buffers[cb])
        submit_info = vk.SubmitInfo(
            s_type=vk.STRUCTURE_TYPE_SUBMIT_INFO,
            wait_dst_stage_mask=pointer(stages),

            # The wait semaphore ensures that the image is presented
		    # before we start submitting command buffers again
            wait_semaphore_count=1,
            wait_semaphores=pointer(self.render_semaphores['present']),

            # The signal semaphore is used during queue presentation
            # to ensure that the image is not rendered before all
            # commands have been submitted
            signal_semaphore_count=1,
            signal_semaphores=pointer(self.render_semaphores['render']),

            # Submit the currently active command buffer
            command_buffer_count=1,
            command_buffers=pointer(drawbuf)
        )

        #Submit to the graphics queue
        assert(self.QueueSubmit(self.queue, 1, byref(submit_info), vk.Fence(0)) == vk.SUCCESS)

        # Present the current buffer to the swap chain
		# We pass the signal semaphore from the submit info
		# to ensure that the image is not rendered until
		# all commands have been submitted
        present_info = vk.PresentInfoKHR(
            s_type=vk.STRUCTURE_TYPE_PRESENT_INFO_KHR, next=None,
            swapchain_count=1, swapchains=pointer(self.swapchain.swapchain),
            image_indices = pointer(current_buffer),
            wait_semaphores = pointer(self.render_semaphores['render']),
            wait_semaphore_count=1
        )

        result = self.QueuePresentKHR(self.queue, byref(present_info))
        if result != vk.SUCCESS:
            raise "Could not render the scene"


    async def render(self):
        """
            Render the scene
        """
        print("Running!")
        import time
        loop = asyncio.get_event_loop()
        frame_counter = 0
        fps_timer = 0.0
        self.running = True

        while self.running:
            t_start = loop.time()

            # draw
            self.DeviceWaitIdle(self.device)
            self.draw()
            self.DeviceWaitIdle(self.device)
            #time.sleep(1/30)

            frame_counter += 1
            t_end = loop.time()
            delta = t_end-t_start
            fps_timer += delta
            if fps_timer > 1:
                self.window.set_title('Texture - {} fps'.format(frame_counter))
                frame_counter = 0
                fps_timer = 0.0
            await asyncio.sleep(0)
        self.rendering_done.set()

    def __init__(self):
        Application.__init__(self)
        self.pipeline_layout = None
        self.pipeline = None
        self.descriptor_set = None
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.render_semaphores = {'present': None, 'render': None}
        self.matrices = (Mat4*3)(Mat4(), Mat4(), Mat4()) # 0: Projection, 1: Model, 2: View

        self.createImageTexture()
        self.imageView = self.createImageView(self.deviceImg, vk.FORMAT_R8G8B8A8_UNORM)
        self.sampler = self.createImageSampler(vk.FILTER_NEAREST, vk.FILTER_NEAREST)
        self.create_semaphores()
        self.create_quad()
        self.create_uniform_buffers()
        self.create_descriptor_set_layout()
        self.create_pipeline()
        self.create_descriptor_pool()
        self.create_descriptor_set()
        self.init_command_buffers()


    def __del__(self):
        self.DestroyDescriptorPool(self.device, self.descriptor_pool, None)
        self.DestroyPipeline(self.device, self.pipeline, None)
        self.DestroyPipelineLayout(self.device, self.pipeline_layout, None)
        self.DestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
        self.DestroyImage(self.device, self.deviceImg, None)
        self.FreeMemory(self.device, self.deviceImgMem, None)
        self.DestroySampler(self.device, self.sampler, None)
        self.DestroyImageView(self.device, self.imageView, None)
        self.DestroyBuffer(self.device, self.vertexBuffer, None)
        self.FreeMemory(self.device, self.vertexBufferMem, None)
        self.DestroyBuffer(self.device, self.indexBuffer, None)
        self.FreeMemory(self.device, self.indexMem, None)
        self.DestroyBuffer(self.device, self.uniformBuffer, None)
        self.FreeMemory(self.device, self.uniformBufferMem, None)

        self.DestroySemaphore(self.device, self.render_semaphores['present'], None)
        self.DestroySemaphore(self.device, self.render_semaphores['render'], None)

        Application.__del__(self)


def main():
    app = TextureApplication()
    app.run()
    loop = asyncio.get_event_loop()
    loop.run_forever()

if __name__ == '__main__':
    main()