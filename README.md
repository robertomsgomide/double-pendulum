# Double Pendulum Simulator

An interactive **Tkinter + Matplotlib** application for studying the nonlinear dynamics and chaotic behaviour of a double pendulum. Two physical models are supported—**simple** (point‐mass bobs) and **compound** (uniform rods)—and the GUI lets you tweak parameters, drag the pendulum into arbitrary initial states, and monitor numerical energy drift in real time.

---

## Contents

| Section                               | Purpose                                           |
| ------------------------------------- | ------------------------------------------------- |
| [Quick start](#quick-start)           | Install dependencies and launch the app           |
| [Features](#features)                 | What the program can do                           |
| [Physics model](#physics-model)       | Equations of motion and energy expressions        |
| [Numerical scheme](#numerical-scheme) | Solver, tolerances, and drift control             |
| [Extending](#extending)               | Where to patch if you need damping, exports, etc. |
| [License](#license)                   | MIT      |

---

## Quick start

```bash
# Clone the repo
$ git clone https://github.com/your‑user/double‑pendulum‑sim.git
$ cd double‑pendulum‑sim

# (Optional but recommended) create a virtual environment
$ python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install runtime dependencies
$ pip install -r requirements.txt

# Run
$ python double_pendulum_sim.py
```

The program opens a resizable window. If you’re on Linux and Tkinter is missing, `sudo apt install python3-tk` usually fixes that.

---

## Features

* **Two physical models** – switch between *simple* (point masses) and *compound* (uniform rods, correct moments of inertia).
* **Parameter panel** – edit lengths $L_1,L_2$, masses $m_1, m_2$, initial angles $\theta_1,\theta_2$, angular velocities, and simulation horizon; *Enter* or focus-out auto-applies.
* **Drag‑and‑drop initial conditions** – click a bob or rod, pull it anywhere on the circle dictated by its length, release, and the solver restarts.
* **Playback controls** – play/pause, single-step &lt;&lt;,&gt;&gt;, jump via time slider, clear traces.
* **Energy monitor** – bottom‑right readout of relative drift $(E(t)-E_0)/E_0$. Useful for judging integrator accuracy.
* **Traces & colours** – each run leaves a faded trajectory, colour‑cycled so comparisons stay readable.
* **Window‑safe** – graceful shutdown closes the Matplotlib backend and frees the Tcl/Tk interpreter.

---

## Physics model

The state vector is

$$
\mathbf y = \begin{bmatrix}\theta_1 & \dot\theta_1 & \theta_2 & \dot\theta_2\end{bmatrix}^T.
$$

### Simple (point masses)

Using Lagrange’s equations with masses *at the rod ends* gives

```math
\begin{aligned}
\dot\theta_1 &= \omega_1,\\[4pt]
\dot\theta_2 &= \omega_2,\\[4pt]
\dot\omega_1 &= \frac{ m_2 L_1 \omega_1^2 \sin\delta\cos\delta + m_2 g \sin\theta_2 \cos\delta + m_2 L_2 \omega_2^2 \sin\delta - (m_1+m_2) g \sin\theta_1 }{(m_1+m_2)L_1 - m_2 L_1\cos^2\delta},\\[6pt]
\dot\omega_2 &= \frac{-(m_1+m_2)(g \sin\theta_2) + (m_1+m_2)(g \sin\theta_1 \cos\delta - L_1 \omega_1^2 \sin\delta) - m_2 L_2\omega_2^2 \sin\delta\cos\delta}{L_2\left(m_1+m_2 - m_2\cos^2\delta\right)},
\end{aligned}
```

where $\delta = \theta_2-\theta_1$.

Total mechanical energy

```math
{\textstyle E = \frac12 m_1 L_1^2 \omega_1^2 + \frac12 m_2\bigl[L_1^2 \omega_1^2 + L_2^2 \omega_2^2 + 2 L_1L_2 \omega_1\omega_2 \cos\delta\bigr] + m_1 g L_1(1-\cos\theta_1) + m_2 g\bigl[L_1(1-\cos\theta_1)+L_2(1-\cos\theta_2)\bigr].}
```

### Compound (uniform rods)

Each rod is a rigid body with centre of mass at $L/2$ and moment of inertia $I=\tfrac13 mL^2$ about its pivot. The mass matrix $\mathbf M$, velocity‑dependent Coriolis vector $\mathbf C$, and gravity vector $\mathbf G$ lead to the compact form

$$
\mathbf M(\theta)\\dot{\boldsymbol\omega} = \mathbf G(\theta) - \mathbf C(\theta,\boldsymbol\omega),
$$

which is solved explicitly for $\dot{\boldsymbol\omega}$ in *double\_pendulum\_deriv*.

---

## Numerical scheme

* **Integrator** – `scipy.integrate.solve_ivp` with Dormand–Prince *(RK45)*.
* **Tolerances** – `rtol=1e-6`, `atol=1e-9`, `max_step=2 ms`. These keep $|\Delta E/E_0|\lesssim10^{-3}$ over 20 s for typical chaotic starts.
* **Energy check** – run `python -c "import double_pendulum_sim as dps; dps.test_energy_drift()"` to see drift statistics.

If you need long‑time symplectic accuracy, swap in a variational integrator or the `DOP853` stepper and tighten tolerances; the architecture is agnostic to solver choice.

---

## Extending

* **Damping** – add terms like `-b1*omega1` inside `double_pendulum_deriv`.
* **Export frames** – Matplotlib’s `FuncAnimation` already handles writers; hook `.save()` in `setup_animation`.
* **Batch runs** – bypass the GUI: import `simulate_double_pendulum()` and sweep parameters.
* **Alternative integrators** – drop‑in replacement in `simulate_double_pendulum()`: try `Radau`, `BDF`, or your own symplectic step.

Pull requests that keep the energy diagnostics, unit tests, and coding conventions are welcome.

---

## License

MIT.  See [`LICENSE`](LICENSE) for the boring legal text.

---
