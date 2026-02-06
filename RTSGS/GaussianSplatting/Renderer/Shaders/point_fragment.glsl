#version 460 core
in vec3 Color;
out vec4 FragColor;
// Hash function for pseudo-random number generation
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main(){
    //make points roundeds
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;

    // Use gl_PrimitiveID to get unique color per point
    float id = float(gl_PrimitiveID);
    
    vec3 color = vec3(
         hash(vec2(id, 0.0)),
         hash(vec2(id, 1.0)),
         hash(vec2(id, 2.0))
    );
    FragColor = vec4(Color.z,Color.y,Color.x, 1.0);
}