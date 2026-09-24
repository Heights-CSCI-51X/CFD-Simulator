import './style.css'

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

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<p> hello </p>
`