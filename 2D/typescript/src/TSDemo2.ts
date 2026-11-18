import Simulator from "./TSSimulator";

const EPSILON = 1e-19;

/**
 * Interface matching the expected structure of your TypeScript Simulator class.
 * Assumes `u_state` is indexed as [row (y)][col (x)][variable], matching NumPy's (nx, ny, 4).
 */

export default class Demo2 {
  private simulator: Simulator;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private imageData: ImageData;

  private initialMaxE: number = 0;
  private initialMinE: number = 0;
  private currentStep: number = 0;
  private isRunning: boolean = false;
  private animationFrameId: number | null = null;

  constructor(
    simulator: Simulator,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D
  ) {
    this.simulator = simulator;
    this.canvas = canvas;
    this.ctx = ctx;

    // Match canvas internal resolution to the grid dimensions.
    // Note: In NumPy (nx, ny), axis 0 (nx) is height (rows) and axis 1 (ny) is width (cols).
    this.canvas.width = this.simulator.ny;
    this.canvas.height = this.simulator.nx;

    // Disable smoothing so pixels stay crisp when scaled up via CSS
    this.ctx.imageSmoothingEnabled = false;

    // Pre-allocate the pixel buffer (RGBA format: 4 bytes per pixel)
    this.imageData = this.ctx.createImageData(
      this.simulator.ny,
      this.simulator.nx
    );

    this.initializeSimulation();
  }

  private initializeSimulation(): void {
    this.simulator.nt = 1_000_000_000;
    this.simulator.update();

    const { min, max } = this.getEnergyExtrema();
    this.initialMaxE = max;
    this.initialMinE = min;
  }

  /**
   * Scans u_state[:, :, 3] to find the minimum and maximum energy values.
   */
  private getEnergyExtrema(): { min: number; max: number } {
    let min = Infinity;
    let max = -Infinity;

    const data = this.simulator.u_state.data as Float64Array; // or Float32Array
    const [strideX, strideY, strideZ] = this.simulator.u_state.strides;

    for (let x = 0; x < this.simulator.nx; x++) {
      const offsetX = x * strideX;
      for (let y = 0; y < this.simulator.ny; y++) {
        const e = data[offsetX + y * strideY + 3 * strideZ];
        if (e < min) min = e;
        if (e > max) max = e;
      }
    }

    return { min, max };
  }

  /**
   * Equivalent to Python's `render(simulator, grid, initial_max_e, initial_min_e)`
   */
  private render(): void {
    const { min: currentMinE } = this.getEnergyExtrema();
    const denominator = this.initialMaxE - this.initialMinE + EPSILON;
    const data = this.imageData.data;

    let pixelIndex = 0;

    const uData = this.simulator.u_state.data;
    const [strideX, strideY, strideC] = this.simulator.u_state.strides;

    for (let x = 0; x < this.simulator.nx; x++) {
      const rowOffset = x * strideX;
      for (let y = 0; y < this.simulator.ny; y++) {
        const e = Number(uData[rowOffset + y * strideY + 3 * strideC]);

        // Normalize and clamp between 0 and 1
        let normalizedE = (e - currentMinE) / denominator;
        if (normalizedE < 0) normalizedE = 0;
        else if (normalizedE > 1) normalizedE = 1;

        // Map [0, 1] to [0, 255] for the Blue channel
        data[pixelIndex] = 0; // R
        data[pixelIndex + 1] = 0; // G
        data[pixelIndex + 2] = Math.round(normalizedE * 255); // B
        data[pixelIndex + 3] = 255; // Alpha

        pixelIndex += 4;
      }
    }

    // Draw the updated pixel buffer onto the canvas
    this.ctx.putImageData(this.imageData, 0, 0);
  }

  /**
   * Starts the main simulation and rendering loop.
   */
  public start(): void {
    if (this.isRunning) return;
    this.isRunning = true;

    const step = () => {
      if (!this.isRunning || this.currentStep >= this.simulator.nt) {
        this.stop();
        return;
      }

      this.simulator.optimizedLf();
      this.render();

      this.currentStep++;
      this.animationFrameId = requestAnimationFrame(step);
    };

    this.animationFrameId = requestAnimationFrame(step);
  }

  /**
   * Halts the simulation loop.
   */
  public stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }
}
