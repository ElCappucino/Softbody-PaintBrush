#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D fluidTexture;

void main() {

    vec3 color = texture(fluidTexture, TexCoords).rgb;
    
    float density = max(color.r, max(color.g, color.b));
    
    if(density < 0.001) discard; 
    
    FragColor = vec4(color, clamp(density, 0.0, 1.0));
}