interface SimulatorOptions {
  x?: number;
  y?: number;
}

export default class Simulator {
  x: number;
  y: number;

  constructor({ x = 0, y = 0 }: SimulatorOptions = {}) {
    this.x = x;
    this.y = y;
  }
}