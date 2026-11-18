import Simulator from "./TSSimulator.ts";
//import Demo from "./TSDemo.ts"
import Demo2 from "./TSDemo2.ts"

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

canvas.width = 20

canvas.height = 20

var my_simulator = new Simulator(canvas.width, canvas.height);

//var my_demo1 = new Demo(my_simulator, canvas)

var my_demo2 = new Demo2(my_simulator, canvas, ctx)

my_demo2.start()

console.log(my_simulator.nx)

  }
}
