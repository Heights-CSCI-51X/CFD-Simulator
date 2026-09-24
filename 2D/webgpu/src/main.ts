import './style.css'
import shadercode from "./computeshader.wgsl?raw"

var canvas = document.getElementById("c") as HTMLCanvasElement
var gpu = undefined

if (!canvas) {
  console.error("No canvas found");
} else {
  gpu = canvas.getContext("webgpu") as RenderingContext
}

if (!gpu) {
  console.error("No WebGPU for you!")
} else {
  console.log("WebGPU found")
}

function makeShader(){

};

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<p> hello </p>
`