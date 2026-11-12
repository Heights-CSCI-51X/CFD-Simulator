import * as np from 'numpy-ts';

// Define constants if not already imported from elsewhere
const EPSILON = 1e-10;
const GAMMA = 1.4;

export class Simulator {
  nx: number;
  ny: number;
  nt: number;

  dx: number;
  dy: number;
  dt: number | null;

  r_start: number;
  r_end: number;
  c_start: number;
  c_end: number;

  u_state: np.NDArray;
  p: np.NDArray;

  f_flux: np.NDArray;
  g_flux: np.NDArray;

  max_speed_x: np.NDArray;
  max_speed_y: np.NDArray;

  a_x: np.NDArray;
  a_y: np.NDArray;

  constructor(nx: number = 8, ny: number = 8) {
    this.nx = nx;
    this.ny = ny;

    this.nt = 3000;

    this.dx = 1 / (this.nx - 1);
    this.dy = 1 / (this.ny - 1);

    this.dt = null;

    this.u_state = np.zeros([this.nx, this.ny, 4]);

    this.r_start = Math.floor(this.ny / 3);
    this.r_end = 2 * Math.floor(this.ny / 3);
    this.c_start = Math.floor(this.nx / 3);
    this.c_end = 2 * Math.floor(this.nx / 3);

    const rho = np.zeros([this.nx, this.ny]).add(EPSILON);

    this.p = np.zeros([this.nx, this.ny]);

    const u = np.zeros([this.nx, this.ny]);
    const v = np.zeros([this.nx, this.ny]);

    rho.fill(0.00000000001);
    u.fill(0.00000000001);
    this.p.fill(0.00000000001);

    rho.slice('0', ':').fill(0.1);
    u.slice('0', ':').fill(2);
    this.p.slice('0', ':').fill(0.1);

    rho.slice('-1', ':').fill(0.1);
    u.slice('-1', ':').fill(-2);
    this.p.slice('-1', ':').fill(0.1);

    const kineticEnergy = rho
      .multiply(0.5)
      .multiply(np.add(np.power(u, 2), np.power(v, 2)));
    const e = this.p.divide(GAMMA - 1).add(kineticEnergy);

    this.setChannel(this.u_state, 0, rho);
    this.setChannel(this.u_state, 1, u.multiply(rho));
    this.setChannel(this.u_state, 2, v.multiply(rho));
    this.setChannel(this.u_state, 3, e);

    this.f_flux = np.zeros([this.nx, this.ny, 4]);
    this.g_flux = np.zeros([this.nx, this.ny, 4]);

    this.max_speed_x = np.zeros([this.nx, this.ny]);
    this.max_speed_y = np.zeros([this.nx, this.ny]);

    this.a_x = np.zeros([this.nx, this.ny]);
    this.a_y = np.zeros([this.nx, this.ny]);
  }

  private setChannel(target: np.NDArray, channel: number, source: np.NDArray): void {
    for (let i = 0; i < this.nx; i++) {
      for (let j = 0; j < this.ny; j++) {
        target.set([i, j, channel], source.item(i, j));
      }
    }
  }
}
