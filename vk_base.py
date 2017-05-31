import platform, asyncio, vk, weakref
from ctypes import (c_void_p, c_float, c_uint8, c_uint, c_uint64, c_int, c_size_t, c_char, c_char_p, cast, Structure, Union, POINTER)
from ctypes import cast, c_char_p, c_uint, c_ubyte, c_ulonglong, pointer, POINTER, byref, c_float, Structure, sizeof, memmove
from itertools import chain

system_name = platform.system()
if system_name == 'Windows':
    from win32 import Win32Window as Window, WinSwapchain as BaseSwapchain
elif system_name == 'Linux':
    from xlib import XlibWindow as Window, XlibSwapchain as BaseSwapchain
else:
    raise OSError("Platform not supported")

# Whether to enable validation layer or not
ENABLE_VALIDATION = False

class Debugger(object):

    def __init__(self, app):
        self.app = weakref.ref(app)
        self.callback_fn = None
        self.debug_report_callback = None

    @staticmethod
    def print_message(flags, object_type, object, location, message_code, layer, message, user_data):
        if flags & vk.DEBUG_REPORT_ERROR_BIT_EXT:
            _type = 'ERROR'
        elif flags & vk.DEBUG_REPORT_WARNING_BIT_EXT:
            _type = 'WARNING'

        print("{}: {}".format(_type, message[::].decode()))
        return 0

    def start(self):
        app = self.app()
        if app is None:
            raise RuntimeError('Application was freed')

        callback_fn = vk.fn_DebugReportCallbackEXT(Debugger.print_message)
        create_info = vk.DebugReportCallbackCreateInfoEXT(
            s_type=vk.STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT,
            next=None,
            flags=vk.DEBUG_REPORT_ERROR_BIT_EXT | vk.DEBUG_REPORT_WARNING_BIT_EXT,
            callback=callback_fn,
            user_data=None
        )

        debug_report_callback = vk.DebugReportCallbackEXT(0)
        result = app.CreateDebugReportCallbackEXT(app.instance, byref(create_info), None, byref(debug_report_callback))

        self.callback_fn = callback_fn
        self.debug_report_callback = debug_report_callback

    def stop(self):
        app = self.app()
        if app is None:
            raise RuntimeError('Application was freed')

        app.DestroyDebugReportCallbackEXT(app.instance, self.debug_report_callback, None)

class Application(object):

    def create_instance(self):
        """
            Setup the vulkan instance
        """
        app_info = vk.ApplicationInfo(
            s_type=vk.STRUCTURE_TYPE_APPLICATION_INFO, next=None,
            application_name=b'PythonText', application_version=0,
            engine_name=b'test', engine_version=0, api_version=vk.API_VERSION_1_0
        )

        if system_name == 'Windows':
            extensions = [b'VK_KHR_surface', b'VK_KHR_win32_surface']
        else:
            extensions = [b'VK_KHR_surface', b'VK_KHR_xcb_surface']

        if ENABLE_VALIDATION:
            extensions.append(b'VK_EXT_debug_report')
            layer_count = 1
            layer_names = [c_char_p(b'VK_LAYER_LUNARG_standard_validation')]
            _layer_names = cast((c_char_p*1)(*layer_names), POINTER(c_char_p))
        else:
            layer_count = 0
            _layer_names = None

        extensions = [c_char_p(x) for x in extensions]
        _extensions = cast((c_char_p*len(extensions))(*extensions), POINTER(c_char_p))

        create_info = vk.InstanceCreateInfo(
            s_type=vk.STRUCTURE_TYPE_INSTANCE_CREATE_INFO, next=None, flags=0,
            application_info=pointer(app_info),

            enabled_layer_count=layer_count,
            enabled_layer_names=_layer_names,

            enabled_extension_count=len(extensions),
            enabled_extension_names=_extensions
        )

        instance = vk.Instance(0)
        result = vk.CreateInstance(byref(create_info), None, byref(instance))
        if result == vk.SUCCESS:
            # For simplicity, all vulkan functions are saved in the application object
            functions = chain(vk.load_functions(instance, vk.InstanceFunctions, vk.GetInstanceProcAddr),
                              vk.load_functions(instance, vk.PhysicalDeviceFunctions, vk.GetInstanceProcAddr))
            for name, function in functions:
                setattr(self, name, function)

            self.instance = instance

            # Start logging errors if validation is enabled
            if ENABLE_VALIDATION:
                self.debugger.start()

        else:
            raise RuntimeError('Instance creation failed. Error code: {}'.format(result))

    def create_device(self):
        self.gpu = None
        self.main_queue_family = None

        # Enumerate the physical devices
        gpu_count = c_uint(0)
        result = self.EnumeratePhysicalDevices(self.instance, byref(gpu_count), None )
        if result != vk.SUCCESS or gpu_count.value == 0:
            raise RuntimeError('Could not fetch the physical devices or there are no devices available')

        buf = (vk.PhysicalDevice*gpu_count.value)()
        self.EnumeratePhysicalDevices(self.instance, byref(gpu_count), cast(buf, POINTER(vk.PhysicalDevice)))


        # For this example use the first available device
        self.gpu = vk.PhysicalDevice(buf[0])

        # Find a graphic queue that supports graphic operation and presentation into
        # the surface previously created
        queue_families_count = c_uint(0)
        self.GetPhysicalDeviceQueueFamilyProperties(
            self.gpu,
            byref(queue_families_count),
            None
        )

        if queue_families_count.value == 0:
            raise RuntimeError('No queues families found for the default GPU')

        queue_families = (vk.QueueFamilyProperties*queue_families_count.value)()
        self.GetPhysicalDeviceQueueFamilyProperties(
            self.gpu,
            byref(queue_families_count),
            cast(queue_families, POINTER(vk.QueueFamilyProperties))
        )

        surface = self.swapchain.surface
        supported = vk.c_uint(0)
        for index, queue in enumerate(queue_families):
            self.GetPhysicalDeviceSurfaceSupportKHR(self.gpu, index, surface, byref(supported))
            if queue.queue_flags & vk.QUEUE_GRAPHICS_BIT != 0 and supported.value == 1:
                self.main_queue_family = index
                break

        if self.main_queue_family is None:
            raise OSError("Could not find a queue that supports graphics and presenting")

        # Create the device
        priorities = (c_float*1)(0.0)
        queue_create_info = vk.DeviceQueueCreateInfo(
            s_type=vk.STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            next=None,
            flags=0,
            queue_family_index=self.main_queue_family,
            queue_count=1,
            queue_priorities=priorities
        )

        queue_create_infos = (vk.DeviceQueueCreateInfo*1)(*(queue_create_info,))

        extensions = (b'VK_KHR_swapchain',)
        _extensions = cast((c_char_p*len(extensions))(*extensions), POINTER(c_char_p))

        if ENABLE_VALIDATION:
            layer_count = 1
            layer_names = (b'VK_LAYER_LUNARG_standard_validation',)
            _layer_names = cast((c_char_p*1)(*layer_names), POINTER(c_char_p))
        else:
            layer_count=0
            _layer_names=None

        create_info = vk.DeviceCreateInfo(
            s_type=vk.STRUCTURE_TYPE_DEVICE_CREATE_INFO, next=None, flags=0,
            queue_create_info_count=1, queue_create_infos=queue_create_infos,

            enabled_layer_count=layer_count,
            enabled_layer_names=_layer_names,

            enabled_extension_count=1,
            enabled_extension_names=_extensions,

            enabled_features=None
        )

        device = vk.Device(0)
        result = self.CreateDevice(self.gpu, byref(create_info), None, byref(device))
        if result == vk.SUCCESS:
            # For simplicity, all vulkan functions are saved in the application object
            # For simplicity, all vulkan functions are saved in the application object
            functions = chain(vk.load_functions(device, vk.QueueFunctions, self.GetDeviceProcAddr),
                              vk.load_functions(device, vk.DeviceFunctions, self.GetDeviceProcAddr),
                              vk.load_functions(device, vk.CommandBufferFunctions, self.GetDeviceProcAddr))

            for name, function in functions:
                setattr(self, name, function)

            self.device = device
        else:
            print(vk.c_int(result))
            raise RuntimeError('Could not create device.')


        # Get the physical device memory properties.
        self.gpu_mem = vk.PhysicalDeviceMemoryProperties()
        self.GetPhysicalDeviceMemoryProperties(self.gpu, byref(self.gpu_mem))

        # Get the queue that was created with the device
        queue = vk.Queue(0)
        self.GetDeviceQueue(device, self.main_queue_family, 0, byref(queue))
        if queue.value != 0:
            self.queue = queue
        else:
            raise RuntimeError("Could not get device queue")

    def create_swapchain(self):
        self.swapchain = Swapchain(self)

    def create_command_pool(self):
        create_info = vk.CommandPoolCreateInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, next= None,
            flags=vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index=self.main_queue_family
        )

        pool = vk.CommandPool(0)
        result = self.CreateCommandPool(self.device, byref(create_info), None, byref(pool))
        if result == vk.SUCCESS:
            self.cmd_pool = pool
        else:
            raise RuntimeError('Could not create command pool')

    def create_setup_buffer(self):
        create_info = vk.CommandBufferAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, next=None,
            command_pool=self.cmd_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count=1
        )
        begin_info = vk.CommandBufferBeginInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            next=None, flags= 0, inheritance_info=None
        )

        if self.setup_buffer is not None:
            self.FreeCommandBuffers(self.device, self.cmd_pool, 1, byref(self.setup_buffer))
            self.setup_buffer = None

        buffer = vk.CommandBuffer(0)
        result = self.AllocateCommandBuffers(self.device, byref(create_info), byref(buffer))
        if result == vk.SUCCESS:
            self.setup_buffer = buffer
        else:
            raise RuntimeError('Failed to create setup buffer')

        if self.BeginCommandBuffer(buffer, byref(begin_info)) != vk.SUCCESS:
            raise RuntimeError('Failed to start recording in the setup buffer')

    def create_command_buffers(self):
        # Create one command buffer per frame buffer
        # in the swap chain
        # Command buffers store a reference to the
        # frame buffer inside their render pass info
        # so for static usage without having to rebuild
        # them each frame, we use one per frame buffer
        image_count = len(self.swapchain.images)
        draw_buffers = (vk.CommandBuffer*image_count)()
        post_present_buffers = (vk.CommandBuffer*image_count)()

        alloc_info = vk.CommandBufferAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, next=None,
            command_pool=self.cmd_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count=image_count
        )

        result = self.AllocateCommandBuffers(self.device, byref(alloc_info), cast(draw_buffers, POINTER(vk.CommandBuffer)))
        if result == vk.SUCCESS:
            self.draw_buffers = draw_buffers
        else:
            raise RuntimeError('Failed to drawing buffers')


        result = self.AllocateCommandBuffers(self.device, byref(alloc_info), cast(post_present_buffers, POINTER(vk.CommandBuffer)))
        if result == vk.SUCCESS:
            self.post_present_buffers = post_present_buffers
        else:
            raise RuntimeError('Failed to present buffers')

    def create2dImage(self, texWidth, texHeight, imgformat, tiling, usage, properties):
        image_info = vk.ImageCreateInfo(
            s_type=vk.STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            next=None,
            flags=0,
            image_type=vk.IMAGE_TYPE_2D,
            format=imgformat,
            tiling=tiling,
            initial_layout=vk.IMAGE_LAYOUT_PREINITIALIZED,
            usage=usage,
            samples=vk.SAMPLE_COUNT_1_BIT,
            sharing_mode=vk.SHARING_MODE_EXCLUSIVE,
            extent=vk.Extent3D(texWidth, texHeight, 1),
            mip_levels=1,
            array_layers=1
        )

        tex_image = vk.Image(0)
        result = self.CreateImage(self.device, byref(image_info), None, byref(tex_image))
        if result != vk.SUCCESS:
            raise RuntimeError('Failed to create texture image')

        memreq = vk.MemoryRequirements()
        self.GetImageMemoryRequirements(self.device, tex_image, byref(memreq))

        memalloc_info = vk.MemoryAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            next=None,
            allocation_size=0,
            memory_type_index=0
        )

        memalloc_info.allocation_size = memreq.size
        memalloc_info.memory_type_index = self.get_memory_type(
            memreq.memory_type_bits,
            vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)[1]

        mem = vk.DeviceMemory(0)
        result = self.AllocateMemory(self.device, byref(memalloc_info), None, byref(mem))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not alloc Memory')

        result = self.BindImageMemory(self.device, tex_image, mem, 0)
        if result != vk.SUCCESS:
            raise RuntimeError('Could not bind the image memory to the image')
        return (tex_image, mem)

    def createBuffer(self, isize, iusage, properties):
        buffer_info = vk.BufferCreateInfo(
            s_type=vk.STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            next=None,
            flags=0,
            size=isize,
            usage=iusage,
            sharing_mode=vk.SHARING_MODE_EXCLUSIVE,
            queue_family_index_count=0,
            queue_family_indices=None
        )
        buffer = vk.Buffer(0)
        result = self.CreateBuffer(self.device, byref(buffer_info), None, byref(buffer))
        if result != vk.SUCCESS:
            raise 'Could not create a buffer'
        memreq = vk.MemoryRequirements()
        self.GetBufferMemoryRequirements(self.device, buffer, byref(memreq))
        memalloc_info = vk.MemoryAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            next=None,
            allocation_size=memreq.size,
            memory_type_index=self.get_memory_type(memreq.memory_type_bits, properties)[1]
        )
        mem = vk.DeviceMemory(0)
        result = self.AllocateMemory(self.device, byref(memalloc_info), None, byref(mem))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not alloc Memory')

        result = self.BindBufferMemory(self.device, buffer, mem, 0)
        return (buffer, mem)

    def beginSingleTimeCommands(self):
        allocInfo = vk.CommandBufferAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            next=None,
            command_pool=self.cmd_pool,
            level=vk.COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count=1)
        commandBuffer = vk.CommandBuffer(0)
        self.AllocateCommandBuffers(self.device, byref(allocInfo), byref(commandBuffer))

        beginInfo = vk.CommandBufferBeginInfo(
            s_type=vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            next=None,
            flags=vk.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            inheritance_info=None)
        result = self.BeginCommandBuffer(commandBuffer, byref(beginInfo))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not begin Command Buffer')
        return commandBuffer

    def endSingleTimeCommands(self, commandBuffer):
        result = self.EndCommandBuffer(commandBuffer)
        assert result == vk.SUCCESS

        submitInfo = vk.SubmitInfo(
            s_type=vk.STRUCTURE_TYPE_SUBMIT_INFO,
            next=None,
            command_buffers=pointer(commandBuffer),
            command_buffer_count=1,
            signal_semaphore_count=0, signal_semaphores=None,
        )
        result = self.QueueSubmit(self.queue, 1, byref(submitInfo), 0)
        assert result == vk.SUCCESS
        result = self.QueueWaitIdle(self.queue)
        assert result == vk.SUCCESS
        self.FreeCommandBuffers(self.device, self.cmd_pool, 1, byref(commandBuffer))


    def copyBuffer(self, srcBuffer, dstBuffer, size):
        cmdBuffer = self.beginSingleTimeCommands()
        copyRegion = vk.BufferCopy(
            size=size
        )
        result = self.CmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, byref(copyRegion))
        endSingleTimeCommands(cmdBuffer)

    def transitionImageLayout(self, image, format, oldLayout, newLayout):
        cmdBuffer = self.beginSingleTimeCommands()

        subres_range = vk.ImageSubresourceRange(
            aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, base_mip_level=0,
            level_count=1, base_array_layer=0, layer_count=1
        )
        barrier = vk.ImageMemoryBarrier(
            s_type=vk.STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            old_layout=oldLayout,
            new_layout=newLayout,
            src_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
            dst_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
            image=image,
            subresource_range=subres_range
        )

        if oldLayout == vk.IMAGE_LAYOUT_PREINITIALIZED and \
        newLayout == vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.srcAccessMask = vk.ACCESS_HOST_WRITE_BIT
            barrier.dstAccessMask = vk.ACCESS_TRANSFER_WRITE_BIT
        elif oldLayout == vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and \
        newLayout == vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = vk.ACCESS_TRANSFER_WRITE_BIT
            barrier.dstAccessMask = vk.ACCESS_SHADER_READ_BIT
        else:
            raise RuntimeError("unsupported layout transition!")


        result = self.CmdPipelineBarrier(
            cmdBuffer,
            vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            byref(barrier))
        self.endSingleTimeCommands(cmdBuffer)

    def uploadToStaging(self, upload_data, size):
        if (self.STAGING_BUFFER_SIZE < size):
            self.DestroyBuffer(self.device, self.stagingBuffer, None)
            self.FreeMemory(self.device, self.stagingMem, None)
            self.createStagingBuffer(size * 2)

        data = vk.c_void_p(0)
        result = self.MapMemory(self.device, self.stagingMem, 0 , size, 0, byref(data))
        if result != vk.SUCCESS:
            raise 'Could not map memory to local'
        memmove(data, upload_data, size)
        self.UnmapMemory(self.device, self.stagingMem)

    def copyBuffer(self, srcBuf, dstBuf, offset, size):
        cmdBuffer = self.beginSingleTimeCommands()
        copyRegion = vk.BufferCopy(
            src_offset=offset,
            dst_offset=0,
            size=size)
        result = self.CmdCopyBuffer(cmdBuffer, srcBuf, dstBuf, 1, byref(copyRegion))

        self.endSingleTimeCommands(cmdBuffer)


    def copyBufferToImage(self, buffer, image, width, height, format):
        self.transitionImageLayout(
            image,
            format,
            vk.IMAGE_LAYOUT_PREINITIALIZED,
            vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            )

        cmdBuffer = self.beginSingleTimeCommands()
        subres_layer = vk.ImageSubresourceLayers(
            aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, mip_level=0,
            base_array_layer=0, layer_count=1
        )
        region = vk.BufferImageCopy(
            buffer_offset=0,
            buffer_row_length=0,
            buffer_image_height=0,
            image_subresource=subres_layer,
            image_offset=vk.Offset3D(0, 0, 0),
            image_extent=vk.Extent3D(width, height, 1)
        )
        result = self.CmdCopyBufferToImage(
            cmdBuffer, buffer, image,
            vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            byref(region))

        self.endSingleTimeCommands(cmdBuffer)
        self.transitionImageLayout(
            image,
            format,
            vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

    def createImageView(self, image, color_format):
        subresource_range = vk.ImageSubresourceRange(
                aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, base_mip_level=0,
                level_count=1, base_array_layer=0, layer_count=1,
            )
        components = vk.ComponentMapping(
            r=vk.COMPONENT_SWIZZLE_R, g=vk.COMPONENT_SWIZZLE_G,
            b=vk.COMPONENT_SWIZZLE_B, a=vk.COMPONENT_SWIZZLE_A
        )

        viewInfo = vk.ImageViewCreateInfo(
            s_type=vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            next=None, flags=0, image=image,
            view_type=vk.IMAGE_VIEW_TYPE_2D, format=color_format,
            components=components, subresource_range=subresource_range
        )
        imageView = vk.ImageView(0)
        result = self.CreateImageView(self.device, byref(viewInfo), None, byref(imageView))
        assert result == vk.SUCCESS
        return imageView

    def createImageSampler(self, magfilter, minfilter):
        samplerInfo = vk.SamplerCreateInfo(
            s_type=vk.STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            next=None,
            mag_filter=magfilter,
            min_filter=minfilter,
            address_mode_u=vk.SAMPLER_ADDRESS_MODE_REPEAT,
            address_mode_v=vk.SAMPLER_ADDRESS_MODE_REPEAT,
            address_mode_w=vk.SAMPLER_ADDRESS_MODE_REPEAT,
            anisotropy_enable=vk.TRUE,
            max_anisotropy=1.0,
            border_color=vk.BORDER_COLOR_INT_OPAQUE_BLACK,
            unnormalized_coordinates=vk.FALSE,
            compare_enable=vk.FALSE,
            compare_op=vk.COMPARE_OP_ALWAYS,
            mipmap_mode=vk.SAMPLER_MIPMAP_MODE_LINEAR,
            mip_lod_bias=0.0,
            min_lod=0.0,
            max_lod=0.0
        )
        imageSampler=vk.Sampler(0)
        result = self.CreateSampler(self.device, pointer(samplerInfo), None, byref(imageSampler))
        assert result == vk.SUCCESS
        return imageSampler

    def createStagingBuffer(self, size):
        self.STAGING_BUFFER_SIZE = size
        (self.stagingBuffer, self.stagingMem ) = self.createBuffer(self.STAGING_BUFFER_SIZE, vk.BUFFER_USAGE_TRANSFER_SRC_BIT, vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT|
        vk.MEMORY_PROPERTY_HOST_COHERENT_BIT)

    def create_depth_stencil(self):
        width, height = self.window.dimensions()

        # Find a supported depth format
        depth_format = None
        depth_formats = (
            vk.FORMAT_D32_SFLOAT_S8_UINT,
            vk.FORMAT_D32_SFLOAT,
            vk.FORMAT_D24_UNORM_S8_UINT,
            vk.FORMAT_D16_UNORM_S8_UINT,
            vk.FORMAT_D16_UNORM,
        )

        format_props = vk.FormatProperties()
        for format in depth_formats:
            self.GetPhysicalDeviceFormatProperties(self.gpu, format, byref(format_props))
            if format_props.optimal_tiling_features & \
            vk.FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT != 0:
                depth_format = format
                break

        if depth_format is None:
            raise RuntimeError('Could not find a valid depth format')

        create_info = vk.ImageCreateInfo(
            s_type=vk.STRUCTURE_TYPE_IMAGE_CREATE_INFO, next=None, flags=0,
            image_type=vk.IMAGE_TYPE_2D, format=depth_format,
            extent=vk.Extent3D(width, height, 1), mip_levels=1,
            array_layers=1, samples=vk.SAMPLE_COUNT_1_BIT, tiling=vk.IMAGE_TILING_OPTIMAL,
            usage=vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | vk.IMAGE_USAGE_TRANSFER_SRC_BIT,
        )

        subres_range = vk.ImageSubresourceRange(
            aspect_mask=vk.IMAGE_ASPECT_DEPTH_BIT, base_mip_level=0,
            level_count=1, base_array_layer=0, layer_count=1,
        )

        create_view_info = vk.ImageViewCreateInfo(
            s_type=vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, next=None,
            flags=0, view_type=vk.IMAGE_VIEW_TYPE_2D, format=depth_format,
            subresource_range=subres_range
        )

        mem_alloc_info = vk.MemoryAllocateInfo(
            s_type=vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, next=None,
            allocation_size=0, memory_type_index=0
        )

        depthstencil_image = vk.Image(0)
        result=self.CreateImage(self.device, byref(create_info), None, byref(depthstencil_image))
        if result != vk.SUCCESS:
            raise RuntimeError('Failed to create depth stencil image')

        memreq = vk.MemoryRequirements()
        self.GetImageMemoryRequirements(self.device, depthstencil_image, byref(memreq))
        mem_alloc_info.allocation_size = memreq.size
        mem_alloc_info.memory_type_index = self.get_memory_type(memreq.memory_type_bits, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)[1]

        depthstencil_mem = vk.DeviceMemory(0)
        result = self.AllocateMemory(self.device, byref(mem_alloc_info), None, byref(depthstencil_mem))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not allocate depth stencil image memory')

        result = self.BindImageMemory(self.device, depthstencil_image, depthstencil_mem, 0)
        if result != vk.SUCCESS:
            raise RuntimeError('Could not bind the depth stencil memory to the image')

        self.set_image_layout(
            self.setup_buffer, depthstencil_image,
            vk.IMAGE_ASPECT_DEPTH_BIT | vk.IMAGE_ASPECT_STENCIL_BIT,
            vk.IMAGE_LAYOUT_UNDEFINED,
            vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        depthstencil_view = vk.ImageView(0)
        create_view_info.image = depthstencil_image
        result = self.CreateImageView(self.device, byref(create_view_info), None, byref(depthstencil_view))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not create image view for depth stencil')

        self.formats['depth'] = depth_format
        self.depth_stencil['image'] = depthstencil_image
        self.depth_stencil['mem'] = depthstencil_mem
        self.depth_stencil['view'] = depthstencil_view

    def create_renderpass(self):
        color, depth = vk.AttachmentDescription(), vk.AttachmentDescription()

        #Color attachment
        color.format = self.formats['color']
        color.samples = vk.SAMPLE_COUNT_1_BIT
        color.load_op = vk.ATTACHMENT_LOAD_OP_CLEAR
        color.store_op = vk.ATTACHMENT_STORE_OP_STORE
        color.stencil_load_op = vk.ATTACHMENT_LOAD_OP_DONT_CARE
        color.stencil_store_op = vk.ATTACHMENT_STORE_OP_DONT_CARE
        color.initial_layout = vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        color.final_layout = vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL

        #Depth attachment
        depth.format = self.formats['depth']
        depth.samples = vk.SAMPLE_COUNT_1_BIT
        depth.load_op = vk.ATTACHMENT_LOAD_OP_CLEAR
        depth.store_op = vk.ATTACHMENT_STORE_OP_STORE
        depth.stencil_load_op = vk.ATTACHMENT_LOAD_OP_DONT_CARE
        depth.stencil_store_op = vk.ATTACHMENT_STORE_OP_DONT_CARE
        depth.initial_layout = vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        depth.final_layout = vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL


        color_ref = vk.AttachmentReference( attachment=0, layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL )
        depth_ref = vk.AttachmentReference( attachment=1, layout=vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL )

        subpass = vk.SubpassDescription(
            pipeline_bind_point = vk.PIPELINE_BIND_POINT_GRAPHICS,
            flags = 0, input_attachment_count=0, input_attachments=None,
            color_attachment_count=1, color_attachments=pointer(color_ref),
            resolve_attachments=None, depth_stencil_attachment=pointer(depth_ref),
            preserve_attachment_count=0, preserve_attachments=None
        )

        attachments = (vk.AttachmentDescription*2)(color, depth)
        create_info = vk.RenderPassCreateInfo(
            s_type=vk.STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            next=None, flags=0, attachment_count=2,
            attachments=cast(attachments, POINTER(vk.AttachmentDescription)),
            subpass_count=1, subpasses=pointer(subpass), dependency_count=0,
            dependencies=None
        )

        renderpass = vk.RenderPass(0)
        result = self.CreateRenderPass(self.device, byref(create_info), None, byref(renderpass))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not create renderpass')

        self.render_pass = renderpass

    def create_pipeline_cache(self):
        create_info = vk.PipelineCacheCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, next=None,
            flags=0, initial_data_size=0, initial_data=None
        )

        pipeline_cache = vk.PipelineCache(0)
        result = self.CreatePipelineCache(self.device, byref(create_info), None, byref(pipeline_cache))
        if result != vk.SUCCESS:
            raise RuntimeError('Failed to create pipeline cache')

        self.pipeline_cache = pipeline_cache

    def create_framebuffers(self):
        attachments = cast((vk.ImageView*2)(), POINTER(vk.ImageView))
        attachments[1] = self.depth_stencil['view']


        width, height = self.window.dimensions()

        create_info = vk.FramebufferCreateInfo(
            s_type=vk.STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            next=None, flags=0, render_pass=self.render_pass,
            attachment_count=2, attachments=attachments,
            width=width, height=height, layers=1
        )

        self.framebuffers = (vk.Framebuffer*len(self.swapchain.images))()
        for index, view in enumerate(self.swapchain.views):
            fb = vk.Framebuffer(0)
            attachments[0] = view

            result = self.CreateFramebuffer(self.device, byref(create_info), None, byref(fb))
            if result != vk.SUCCESS:
                raise RuntimeError('Could not create the framebuffers')

            self.framebuffers[index] = fb

    def flush_setup_buffer(self):
        if self.EndCommandBuffer(self.setup_buffer) != vk.SUCCESS:
            raise RuntimeError('Failed to end setup command buffer')

        submit_info = vk.SubmitInfo(
            s_type=vk.STRUCTURE_TYPE_SUBMIT_INFO, next=None,
            wait_semaphore_count=0, wait_semaphores=None,
            wait_dst_stage_mask=None, command_buffer_count=1,
            command_buffers=pointer(self.setup_buffer),
            signal_semaphore_count=0, signal_semaphores=None,
        )

        result = self.QueueSubmit(self.queue, 1, byref(submit_info), 0)
        if result != vk.SUCCESS:
            raise RuntimeError("Setup buffer sumbit failed")

        result = self.QueueWaitIdle(self.queue)
        if result != vk.SUCCESS:
            raise RuntimeError("Setup execution failed")

        self.FreeCommandBuffers(self.device, self.cmd_pool, 1, byref(self.setup_buffer))
        self.setup_buffer = None

    def set_image_layout(self, cmd, image, aspect_mask, old_layout, new_layout, subres=None):

        if subres is None:
            subres = vk.ImageSubresourceRange(
                aspect_mask=aspect_mask, base_mip_level=0,
                level_count=1, base_array_layer=0, layer_count=1,
            )

        barrier = vk.ImageMemoryBarrier(
            s_type=vk.STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, next=None,
            old_layout=old_layout, new_layout=new_layout,
            src_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
            dst_queue_family_index=vk.QUEUE_FAMILY_IGNORED,
            image=image, subresource_range=subres
        )

        # Source layouts mapping (old)
        old_map = {
            vk.IMAGE_LAYOUT_PREINITIALIZED: vk.ACCESS_HOST_WRITE_BIT | vk.ACCESS_TRANSFER_WRITE_BIT,
            vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: vk.ACCESS_TRANSFER_READ_BIT,
            vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: vk.ACCESS_SHADER_READ_BIT
        }
        if old_layout in old_map.values():
            barrier.src_access_mask = old_map[old_layout]
        else:
            barrier.src_access_mask = 0

        # Target layouts
        if new_layout == vk.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            barrier.dst_access_mask = vk.ACCESS_TRANSFER_WRITE_BIT

        elif new_layout == vk.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            barrier.src_access_mask |= vk.ACCESS_TRANSFER_READ_BIT
            barrier.dst_access_mask = vk.ACCESS_TRANSFER_READ_BIT

        elif new_layout == vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            barrier.dst_access_mask = vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            barrier.src_access_mask = vk.ACCESS_TRANSFER_READ_BIT

        elif new_layout == vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            barrier.dst_access_mask |= vk.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT

        elif new_layout == vk.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.src_access_mask = vk.ACCESS_HOST_WRITE_BIT | vk.ACCESS_TRANSFER_WRITE_BIT
            barrier.dst_access_mask = vk.ACCESS_SHADER_READ_BIT

        self.CmdPipelineBarrier(
            cmd,
            vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0,
            0,None,
            0,None,
            1, byref(barrier)
        )

    def get_memory_type(self, bits, properties):
        for index, mem_t in enumerate(self.gpu_mem.memory_types):
            if (bits & 1) == 1:
                if mem_t.property_flags & properties == properties:
                    return (True, index)
            bits >>= 1

        return (False, None)

    def load_shader(self, name, stage):
        # Read the shader data
        path = './shaders/{}'.format(name)
        shader_f = open(path, 'rb')
        shader_bin = shader_f.read()
        shader_bin_size = len(shader_bin)
        shader_bin = (c_ubyte*shader_bin_size)(*shader_bin)
        shader_f.close()

        # Compile the shader
        module = vk.ShaderModule(0)
        module_create_info = vk.ShaderModuleCreateInfo(
            s_type=vk.STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, next=None,
            code_size=len(shader_bin), code=cast(shader_bin, POINTER(c_uint))
        )

        result = self.CreateShaderModule(self.device, byref(module_create_info), None, byref(module))
        if result != vk.SUCCESS:
            raise RuntimeError('Could not compile shader at {}'.format(path))

        shader_info = vk.PipelineShaderStageCreateInfo(
            s_type=vk.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, next=None,
            flags=0, stage=stage, module=module, name=b'main',
            specialization_info=None
        )

        self.shaders_modules.append(module)
        return shader_info

    def resize_display(self, width, height):
        if not self.initialized:
            return

        self.create_setup_buffer()

        # Recreate the swap chain
        self.swapchain.create()

        # Recreate the frame buffers
        self.DestroyImageView(self.device, self.depth_stencil['view'], None)
        self.DestroyImage(self.device, self.depth_stencil['image'], None)
        self.FreeMemory(self.device, self.depth_stencil['mem'], None)
        self.create_depth_stencil()

        for fb in self.framebuffers:
            self.DestroyFramebuffer(self.device, fb, None)
        self.create_framebuffers()

        self.flush_setup_buffer()

        # Command buffers need to be recreated as they may store
	    # references to the recreated frame buffer
        len_draw_buffers = len(self.draw_buffers)
        self.FreeCommandBuffers(self.device, self.cmd_pool, len_draw_buffers, cast(self.draw_buffers, POINTER(vk.CommandBuffer)))
        self.FreeCommandBuffers(self.device, self.cmd_pool, len_draw_buffers, cast(self.post_present_buffers, POINTER(vk.CommandBuffer)))
        self.create_command_buffers()

    def __init__(self):
        self.initialized = False
        self.running = False
        self.zoom = -2.5               # Scene zoom
        self.rotation = (c_float*3)()  # Scene rotation
        self.shaders_modules = []      # A list of compiled shaders. GC'ed with the application
        self.debugger = Debugger(self) # Throw errors if validations layers are activated

        # Syncronization between the system events and the rendering
        self.rendering_done = asyncio.Event()

        #System window
        self.window = Window(self)

        # Vulkan objets
        self.gpu = None
        self.gpu_mem = None
        self.instance = None
        self.device = None
        self.queue = None
        self.swapchain = None
        self.cmd_pool = None
        self.setup_buffer = None
        self.draw_buffers = []
        self.post_present_buffers = []
        self.render_pass = None
        self.pipeline_cache = None
        self.framebuffers = None
        self.depth_stencil = {'image':None, 'mem':None, 'view':None}
        self.formats = {'color':None, 'depth':None}

        # Vulkan objets initialization
        self.create_instance()
        self.create_swapchain()
        self.create_device()
        self.create_command_pool()

        self.create_setup_buffer()
        self.swapchain.create()
        self.create_command_buffers()
        self.create_depth_stencil()
        self.create_renderpass()
        self.create_pipeline_cache()
        self.create_framebuffers()
        self.flush_setup_buffer()
        self.createStagingBuffer(8192)

        self.window.show()

    def __del__(self):
        if self.instance is None:
            return

        dev = self.device
        if dev is not None:
            self.DestroyBuffer(self.device, self.stagingBuffer, None)
            self.FreeMemory(self.device, self.stagingMem, None)

            if self.swapchain is not None:
                self.swapchain.destroy()

            if self.setup_buffer is not None:
                self.FreeCommandBuffers(dev, self.cmd_pool, 1, byref(self.setup_buffer))

            len_draw_buffers = len(self.draw_buffers)
            if len_draw_buffers > 0:
                self.FreeCommandBuffers(dev, self.cmd_pool, len_draw_buffers, cast(self.draw_buffers, POINTER(vk.CommandBuffer)))
                self.FreeCommandBuffers(dev, self.cmd_pool, len_draw_buffers, cast(self.post_present_buffers, POINTER(vk.CommandBuffer)))

            if self.render_pass is not None:
                self.DestroyRenderPass(self.device, self.render_pass, None)

            for mod in self.shaders_modules:
                self.DestroyShaderModule(self.device, mod, None)

            if self.framebuffers is not None:
                for fb in self.framebuffers:
                    self.DestroyFramebuffer(self.device, fb, None)

            if self.depth_stencil['view'] is not None:
                self.DestroyImageView(dev, self.depth_stencil['view'], None)

            if self.depth_stencil['image'] is not None:
                self.DestroyImage(dev, self.depth_stencil['image'], None)

            if self.depth_stencil['mem'] is not None:
                self.FreeMemory(dev, self.depth_stencil['mem'], None)

            if self.pipeline_cache:
                self.DestroyPipelineCache(self.device, self.pipeline_cache, None)

            if self.cmd_pool:
                self.DestroyCommandPool(dev, self.cmd_pool, None)

            self.DestroyDevice(dev, None)

        if ENABLE_VALIDATION:
            self.debugger.stop()

        self.DestroyInstance(self.instance, None)
        print('Application freed!')

class Swapchain(BaseSwapchain):

    def __init__(self, app):
        super().__init__(app)

        self.swapchain = None
        self.images = None
        self.views = None

    def create(self):
        app = self.app()

        # Get the physical device surface capabilities (properties and format)
        cap = vk.SurfaceCapabilitiesKHR()
        result = app.GetPhysicalDeviceSurfaceCapabilitiesKHR(app.gpu, self.surface, byref(cap))
        if result != vk.SUCCESS:
            raise RuntimeError('Failed to get surface capabilities')

        # Get the available present mode
        prez_count = c_uint(0)
        result = app.GetPhysicalDeviceSurfacePresentModesKHR(app.gpu, self.surface, byref(prez_count), None)
        if result != vk.SUCCESS and prez_count.value > 0:
            raise RuntimeError('Failed to get surface presenting mode')

        prez = (c_uint*prez_count.value)()
        app.GetPhysicalDeviceSurfacePresentModesKHR(app.gpu, self.surface, byref(prez_count), cast(prez, POINTER(c_uint)) )

        if cap.current_extent.width == -1:
            # If the surface size is undefined, the size is set to the size of the images requested
            width, height = app.window.dimensions()
            swapchain_extent = vk.Extent2D(width=width, height=height)
        else:
            # If the surface size is defined, the swap chain size must match
            # The client most likely uses windowed mode
            swapchain_extent = cap.current_extent
            width = swapchain_extent.width
            height = swapchain_extent.height

        # Prefer mailbox mode if present, it's the lowest latency non-tearing present  mode
        present_mode = vk.PRESENT_MODE_FIFO_KHR
        if vk.PRESENT_MODE_MAILBOX_KHR in prez:
            present_mode = vk.PRESENT_MODE_MAILBOX_KHR
        elif vk.PRESENT_MODE_IMMEDIATE_KHR in prez:
            present_mode = vk.PRESENT_MODE_IMMEDIATE_KHR

        # Get the number of images
        swapchain_image_count = cap.min_image_count + 1
        if cap.max_image_count > 0 and swapchain_image_count > cap.max_image_count:
            swapchain_image_count = cap.max_image_count

        # Default image transformation (use identity if supported)
        transform = cap.current_transform
        if cap.supported_transforms & vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR != 0:
            transform = vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR

        # Get the supported image format
        format_count = c_uint(0)
        result = app.GetPhysicalDeviceSurfaceFormatsKHR(app.gpu, self.surface, byref(format_count), None)
        if result != vk.SUCCESS and format_count.value > 0:
            raise RuntimeError('Failed to get surface available image format')

        formats = (vk.SurfaceFormatKHR*format_count.value)()
        app.GetPhysicalDeviceSurfaceFormatsKHR(app.gpu, self.surface, byref(format_count), cast(formats, POINTER(vk.SurfaceFormatKHR)))

        # If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
		# there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
        if format_count == 1 and formats[0].format == vk.FORMAT_UNDEFINED:
            color_format = vk.FORMAT_B8G8R8A8_UNORM
        else:
            # Else select the first format
            color_format = formats[0].format

        app.formats['color'] = color_format
        color_space = formats[0].color_space

        #Create the swapchain
        create_info = vk.SwapchainCreateInfoKHR(
            s_type=vk.STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, next=None,
            flags=0, surface=self.surface, min_image_count=swapchain_image_count,
            image_format=color_format, image_color_space=color_space,
            image_extent=swapchain_extent, image_array_layers=1, image_usage=vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            image_sharing_mode=vk.SHARING_MODE_EXCLUSIVE, queue_family_index_count=0,
            queue_family_indices=cast(None, POINTER(c_uint)), pre_transform=transform,
            composite_alpha=vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR, present_mode=present_mode,
            clipped=1,
            old_swapchain=(self.swapchain or vk.SwapchainKHR(0))
        )

        swapchain = vk.SwapchainKHR(0)
        result = app.CreateSwapchainKHR(app.device, byref(create_info), None, byref(swapchain))

        if result == vk.SUCCESS:
            if self.swapchain is not None: #Destroy the old swapchain if it exists
                self.destroy_swapchain()
            self.swapchain = swapchain
            self.create_images(swapchain_image_count, color_format)
        else:
            raise RuntimeError('Failed to create the swapchain')

    def create_images(self, req_image_count, color_format):
        app = self.app()

        image_count = c_uint(0)
        result = app.GetSwapchainImagesKHR(app.device, self.swapchain, byref(image_count), None)
        if result != vk.SUCCESS and req_image_count != image_count.value:
            raise RuntimeError('Failed to get the swapchain images')

        self.images = (vk.Image * image_count.value)()
        self.views = (vk.ImageView * image_count.value)()

        assert( app.GetSwapchainImagesKHR(app.device, self.swapchain, byref(image_count), cast(self.images, POINTER(vk.Image))) == vk.SUCCESS)

        for index, image in enumerate(self.images):
            components = vk.ComponentMapping(
                r=vk.COMPONENT_SWIZZLE_R, g=vk.COMPONENT_SWIZZLE_G,
                b=vk.COMPONENT_SWIZZLE_B, a=vk.COMPONENT_SWIZZLE_A,
            )

            subresource_range = vk.ImageSubresourceRange(
                aspect_mask=vk.IMAGE_ASPECT_COLOR_BIT, base_mip_level=0,
                level_count=1, base_array_layer=0, layer_count=1,
            )

            view_create_info = vk.ImageViewCreateInfo(
                s_type=vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                next=None, flags=0, image=image,
                view_type=vk.IMAGE_VIEW_TYPE_2D, format=color_format,
                components=components, subresource_range=subresource_range
            )

            app.set_image_layout(
                app.setup_buffer, image,
                vk.IMAGE_ASPECT_COLOR_BIT,
                vk.IMAGE_LAYOUT_UNDEFINED,
                vk.IMAGE_LAYOUT_PRESENT_SRC_KHR)

            view = vk.ImageView(0)
            result = app.CreateImageView(app.device, byref(view_create_info), None, byref(view))
            if result == vk.SUCCESS:
                self.views[index] = view
            else:
                raise RuntimeError('Failed to create an image view.')

    def destroy_swapchain(self):
        app = self.app()
        for view in self.views:
            app.DestroyImageView(app.device, view, None)
        app.DestroySwapchainKHR(app.device, self.swapchain, None)

    def destroy(self):
        app = self.app()
        if self.swapchain is not None:
            self.destroy_swapchain()
        app.DestroySurfaceKHR(app.instance, self.surface, None)
