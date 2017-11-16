#version 450

layout (vertices = 4) out;
layout (location = 0) in vec2 texCoord[];
layout (location = 0) out vec4 OUT[];
layout (location = 1) out vec2 outTex[];

void main()
{

    gl_TessLevelInner[0] = 5.0;
    gl_TessLevelInner[1] = 6.0;

    gl_TessLevelOuter[0] = 1.0;
    gl_TessLevelOuter[1] = 2.0;
    gl_TessLevelOuter[2] = 3.0;
    gl_TessLevelOuter[3] = 4.0;

   OUT[gl_InvocationID] = gl_in[gl_InvocationID].gl_Position;
   outTex[gl_InvocationID] = texCoord[gl_InvocationID];
}