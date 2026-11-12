import Simulator from "./TSSimulator.ts";

const canvas = document.getElementById("c");

if (!(canvas instanceof HTMLCanvasElement) || (canvas == null)) {
  console.error("No canvas for you!");
} else {
  console.log("Canvas has been located");
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    console.error("No context for rendering :(")
  } else {
    console.log("Context has been found!")

var my_simulator = new Simulator({ 
  x: canvas.clientWidth, 
  y: canvas.clientHeight 
});

console.log(my_simulator.x)

  }
}
