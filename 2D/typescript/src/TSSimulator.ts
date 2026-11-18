import * as np from 'numpy-ts';

// Define constants if not already imported from elsewhere
const GAMMA = 1.4;
const CFL = 0.5;
const EPSILON = 1e-8;

export default class Simulator {
  nx: number;
  ny: number;
  nt: number;

  dx: number;
  dy: number;
  dt: number;

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

    this.dt = EPSILON

    this.u_state = np.zeros([this.nx, this.ny, 4]);

    this.r_start = Math.floor(this.ny / 3);
    this.r_end = 2 * Math.floor(this.ny / 3);
    this.c_start = Math.floor(this.nx / 3);
    this.c_end = 2 * Math.floor(this.nx / 3);

    const rho = np.zeros([this.nx, this.ny]).add(EPSILON);

    this.p = np.zeros([this.nx, this.ny]);

    const u = np.zeros([this.nx, this.ny]);
    const v = np.zeros([this.nx, this.ny]);

    rho.fill(0.0001);
    u.fill(0.00000000001);
    this.p.fill(0.0001);

    rho.slice('0', ':').fill(0.001);
    u.slice('0', ':').fill(0.00001);
    this.p.slice('0', ':').fill(0.01);

    rho.slice('-1', ':').fill(0.001);
    u.slice('-1', ':').fill(-0.00001);
    this.p.slice('-1', ':').fill(0.01);

    const kineticEnergy = rho
      .multiply(0.5)
      .multiply(np.add(np.power(u, 2), np.power(v, 2)));
    const e = this.p.divide(GAMMA - 1).add(kineticEnergy);

    this.setChannel(this.u_state, 0, rho);
    this.setChannel(this.u_state, 1, u.multiply(rho));
    this.setChannel(this.u_state, 2, v.multiply(rho));
    this.setChannel(this.u_state, 3, e);

    rho.dispose()
    u.dispose()
    v.dispose()
    e.dispose()

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

  updateFlux(): void {
    let rho = this.u_state.slice(':', ':', '0');
    rho = np.maximum(rho, EPSILON);

    const u = this.u_state.slice(':', ':', '1').divide(rho);
    const v = this.u_state.slice(':', ':', '2').divide(rho);
    const e = this.u_state.slice(':', ':', '3');

    const rhoU = rho.multiply(u);
    const rhoV = rho.multiply(v);
    const rhoUV = rhoU.multiply(v);
    const ePlusP = e.add(this.p);

    const f0 = rhoU;
    const f1 = rho.multiply(u.power(2)).add(this.p);
    const f2 = rhoUV;
    const f3 = ePlusP.multiply(u);
    this.f_flux = np.stack([f0, f1, f2, f3], -1);

    const g0 = rhoV;
    const g1 = rhoUV;
    const g2 = rho.multiply(v.power(2)).add(this.p);
    const g3 = ePlusP.multiply(v);
    this.g_flux = np.stack([g0, g1, g2, g3], -1);
  }

  updateMaxSpeed(): void {
    let rho = this.u_state.slice(':', ':', '0');
    rho = np.maximum(rho, EPSILON);

    const u = np.divide(this.u_state.slice(':', ':', '1'), rho);
    const v = np.divide(this.u_state.slice(':', ':', '2'), rho);

    const pSafe = np.abs(this.p);
    const rhoSafe = np.abs(rho);
    const c = np.sqrt(np.divide(np.multiply(pSafe, GAMMA), rhoSafe));

    this.max_speed_x = np.add(np.abs(u), c);
    this.max_speed_y = np.add(np.abs(v), c);
  }
  public updateA(): void {
    this.a_x = np.maximum(
      this.max_speed_x.slice(':-1', ':'),
      this.max_speed_x.slice('1:', ':')
    );

    this.a_y = np.maximum(
      this.max_speed_y.slice(':', ':-1'),
      this.max_speed_y.slice(':', '1:')
    );
  }

  public updateDt(): void {
    const a_x_max = np.max(this.max_speed_x.slice('1:-1', '1:-1')) as number;
    const a_y_max = np.max(this.max_speed_y.slice('1:-1', '1:-1')) as number;

    const dt = CFL / ((a_x_max / this.dx) + (a_y_max / this.dy) + EPSILON);
    this.dt = Math.min(dt, 1);
  }

  public updateP(): void {
    const rho = this.u_state.slice(':', ':', '0');
    const mom_x = this.u_state.slice(':', ':', '1');
    const mom_y = this.u_state.slice(':', ':', '2');
    const E = this.u_state.slice(':', ':', '3');

    const kinetic_energy = mom_x
      .power(2)
      .add(mom_y.power(2))
      .multiply(0.5)
      .divide(rho.add(EPSILON));

    this.p = E.subtract(kinetic_energy).multiply(GAMMA - 1);
  }

  public updateE(): void {
    let rho = this.u_state.slice(':', ':', '0');
    rho = np.maximum(rho, EPSILON);

    const u = this.u_state.slice(':', ':', '1').divide(rho);
    const v = this.u_state.slice(':', ':', '2').divide(rho);

    const internal_e = this.p.divide(GAMMA - 1);
    const kinetic_e = u
      .power(2)
      .add(v.power(2))
      .multiply(rho)
      .multiply(0.5);

    const new_e = internal_e.add(kinetic_e);

    this.u_state = np.stack(
      [
        this.u_state.slice(':', ':', '0'),
        this.u_state.slice(':', ':', '1'),
        this.u_state.slice(':', ':', '2'),
        new_e,
      ],
      2
    );
  }

  public update(): void {
    this.updateP();
    this.updateE();
    this.updateFlux();
    this.updateMaxSpeed();
    this.updateA();
    this.updateDt();
  }

  optimizedLf(): void {
    this.update();

    const f_avg = this.f_flux
      .slice(':-1', ':', ':')
      .add(this.f_flux.slice('1:', ':', ':'))
      .multiply(0.5);

    const ax_expanded = this.a_x.slice(':', ':', 'newaxis');
    const u_diff_x = this.u_state
      .slice('1:', ':', ':')
      .subtract(this.u_state.slice(':-1', ':', ':'));

    const flux_at_the_walls_x = f_avg.subtract(
      ax_expanded.multiply(0.5).multiply(u_diff_x)
    );

    const g_avg = this.g_flux
      .slice(':', ':-1', ':')
      .add(this.g_flux.slice(':', '1:', ':'))
      .multiply(0.5);

    const ay_expanded = this.a_y.slice(':', ':', 'newaxis');
    const u_diff_y = this.u_state
      .slice(':', '1:', ':')
      .subtract(this.u_state.slice(':', ':-1', ':'));

    const flux_at_the_walls_y = g_avg.subtract(
      ay_expanded.multiply(0.5).multiply(u_diff_y)
    );

    const net_flux_x = flux_at_the_walls_x
      .slice('1:', '1:-1', ':')
      .subtract(flux_at_the_walls_x.slice(':-1', '1:-1', ':'))
      .multiply(this.dt / this.dx);

    const net_flux_y = flux_at_the_walls_y
      .slice('1:-1', '1:', ':')
      .subtract(flux_at_the_walls_y.slice('1:-1', ':-1', ':'))
      .multiply(this.dt / this.dy);

    const interior_updated = this.u_state
      .slice('1:-1', '1:-1', ':')
      .subtract(net_flux_x.add(net_flux_y));

    np.copyto(this.u_state.slice('1:-1', '1:-1', ':'), interior_updated)

    np.copyto(this.u_state.slice('0', ':', ':'), this.u_state.slice('1', ':', ':'))
    np.copyto(this.u_state.slice('-1', ':', ':'), this.u_state.slice('-2', ':', ':'))
    np.copyto(this.u_state.slice(':', '0', ':'), this.u_state.slice(':', '1', ':'))
    np.copyto(this.u_state.slice(':', '-1', ':'), this.u_state.slice(':', '-2', ':'))

    ax_expanded.dispose()
    ay_expanded.dispose()
    interior_updated.dispose()
    u_diff_x.dispose()
    u_diff_y.dispose()
    net_flux_x.dispose()
    net_flux_y.dispose()
    flux_at_the_walls_x.dispose()
    flux_at_the_walls_y.dispose()
    f_avg.dispose()
  }
}
