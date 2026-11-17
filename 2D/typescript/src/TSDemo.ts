//Got this from AI, plan to modifiy it, just wanted quick renderer to make sure simulator was working --> it is hard to visulaze in 2D

// Adjust this interface to match your actual Simulator class and tensor library (e.g., numjs / ndarray)
export interface Tensor3D {
  shape: number[];
  get(i: number, j: number, k: number): number;
  data?: Float32Array | Float64Array | number[];
}

export interface FluidSimulator {
  nx: number;
  ny: number;
  u_state: Tensor3D;
  step?: () => void; // Optional: include if you want Demo to drive the animation loop
}

export type VisualizationChannel = 'density' | 'pressure' | 'velocity' | 'energy';

export class Demo {
  private sim: FluidSimulator;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  // Offscreen buffer matching the exact grid dimensions (nx × ny)
  private offscreenCanvas: HTMLCanvasElement;
  private offscreenCtx: CanvasRenderingContext2D;
  private imageData: ImageData;

  private channel: VisualizationChannel = 'density';
  private gamma: number = 1.4;
  private animationFrameId: number | null = null;

  constructor(sim: FluidSimulator, canvas: HTMLCanvasElement, gamma: number = 1.4) {
    this.sim = sim;
    this.canvas = canvas;
    this.gamma = gamma;

    const context = this.canvas.getContext('2d');
    if (!context) {
      throw new Error('Could not initialize 2D context from the provided canvas.');
    }
    this.ctx = context;

    // Create an offscreen canvas at simulation resolution for fast pixel manipulation
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = this.sim.nx;
    this.offscreenCanvas.height = this.sim.ny;

    const offCtx = this.offscreenCanvas.getContext('2d');
    if (!offCtx) {
      throw new Error('Could not initialize offscreen 2D context.');
    }
    this.offscreenCtx = offCtx;
    this.imageData = this.offscreenCtx.createImageData(this.sim.nx, this.sim.ny);
  }

  /**
   * Switch which physical property is rendered.
   */
  public setChannel(channel: VisualizationChannel): void {
    this.channel = channel;
  }

  /**
   * Renders the current simulation state onto the canvas.
   */
  public render(): void {
    const { nx, ny, u_state } = this.sim;
    const pixels = this.imageData.data;

    // First pass: extract scalar values and find min/max for dynamic color normalization
    const field = new Float32Array(nx * ny);
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        // Channel 0: rho, Channel 1: rho*u, Channel 2: rho*v, Channel 3: total energy E
        const rho = Math.max(u_state.get(i, j, 0), 1e-12);
        const rhou = u_state.get(i, j, 1);
        const rhov = u_state.get(i, j, 2);
        const E = u_state.get(i, j, 3);

        let value = rho;

        if (this.channel === 'velocity') {
          const u = rhou / rho;
          const v = rhov / rho;
          value = Math.hypot(u, v);
        } else if (this.channel === 'pressure') {
          const u = rhou / rho;
          const v = rhov / rho;
          const kinetic = 0.5 * rho * (u * u + v * v);
          value = Math.max((this.gamma - 1) * (E - kinetic), 0);
        } else if (this.channel === 'energy') {
          value = E;
        }

        const idx = j * nx + i; // Map x -> column (i), y -> row (j)
        field[idx] = value;

        if (value < minVal) minVal = value;
        if (value > maxVal) maxVal = value;
      }
    }

    const range = maxVal - minVal || 1;

    // Second pass: write RGBA values into ImageData
    for (let idx = 0; idx < field.length; idx++) {
      const normalized = (field[idx] - minVal) / range;
      const [r, g, b] = this.colormap(normalized);

      const pixelOffset = idx * 4;
      pixels[pixelOffset] = r;
      pixels[pixelOffset + 1] = g;
      pixels[pixelOffset + 2] = b;
      pixels[pixelOffset + 3] = 255;
    }

    // Draw pixel buffer to offscreen canvas, then scale up to main canvas
    this.offscreenCtx.putImageData(this.imageData, 0, 0);

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.imageSmoothingEnabled = false; // Crisp grid cells (set true for bilinear blur)
    this.ctx.drawImage(
      this.offscreenCanvas,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );
  }

  /**
   * Optional animation loop if your simulator exposes a `step()` method.
   */
  public start(stepsPerFrame: number = 1): void {
    if (this.animationFrameId !== null) return;

    const loop = () => {
      if (this.sim.step) {
        for (let s = 0; s < stepsPerFrame; s++) {
          this.sim.step();
        }
      }
      this.render();
      this.animationFrameId = requestAnimationFrame(loop);
    };

    this.animationFrameId = requestAnimationFrame(loop);
  }

  public stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Maps a [0, 1] value to a heatmap RGB color (dark blue -> cyan -> yellow -> red).
   */
  private colormap(t: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * clamped - 3))));
    const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * clamped - 2))));
    const b = Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * clamped - 1))));
    return [r, g, b];
  }
}
