#version 450

layout (quads, equal_spacing, ccw) in;
layout(location = 0) in vec2 outTex[];
layout (location = 1) in vec4 OUT[];
layout(location = 0) out vec2 fragTexCoord;
void main()
{
    float u = gl_TessCoord.x;
    float omu = 1 - u;
    float v = gl_TessCoord.y;
    float omv = 1 - v;

    float p0 = omu * omv;
    float p1 = u * omv;
    float p2 = u * v;
    float p3 = omu * v;

    gl_Position = p0 * OUT[0] +
                  p1 * OUT[1] +
                  p2 * OUT[2] +
                  p3 * OUT[3];
    fragTexCoord = p0 * outTex[0] + p1 * outTex[1] + p2 * outTex[2]  + p3 * outTex[3];
}