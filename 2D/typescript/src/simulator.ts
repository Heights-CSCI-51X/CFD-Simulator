interface SimulatorOptions {
  x?: number;
  y?: number;
}

class Simulator {
  x: number;
  y: number;

  constructor({ x = 0, y = 0 }: SimulatorOptions = {}) {
    this.x = x;
    this.y = y;
  }
}

const simulator = new Simulator();
console.log(simulator.x); // 0
