#version 330 core
out vec4 FragColor;

uniform vec3 objectColor; // New uniform for custom colors

void main() {
    FragColor = vec4(objectColor, 1.0);
}