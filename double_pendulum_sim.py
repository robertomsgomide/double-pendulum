import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
import random
import tkinter as tk
from tkinter import ttk, messagebox

def double_pendulum_deriv(t, y, L1, L2, m1, m2, g=9.81, pendulum_type='simple'):
    """
    Compute the derivatives of the double pendulum's state variables.
    
    State variables:
    y = [theta1, omega1, theta2, omega2]
    
    Parameters:
    t - time (not used, but required by the ODE solver)
    y - state vector [theta1, omega1, theta2, omega2]
    L1, L2 - lengths of the pendulum arms
    m1, m2 - masses
    g - gravitational acceleration
    pendulum_type - 'simple' (point mass) or 'compound' (distributed mass)
    """
    theta1, omega1, theta2, omega2 = y
    
    if pendulum_type == 'simple':
        # Equations for a simple double pendulum (point masses)
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
        den2 = L2 * (m1 + m2 - m2 * np.cos(delta) * np.cos(delta))
        
        # Derivatives
        dtheta1 = omega1
        dtheta2 = omega2
        
        # Calculate omega1 and omega2 derivatives
        domega1 = (m2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * L2 * omega2 * omega2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1
        
        domega2 = (-m2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) - 
                              L1 * omega1 * omega1 * np.sin(delta) - 
                              g * np.sin(theta2))) / den2
    else:
        # For uniform rods, center of mass is at L/2
        Lc1 = L1 / 2  # Center of mass of rod 1
        Lc2 = L2 / 2  # Center of mass of rod 2
        
        # Moments of inertia about pivot points
        I1 = (1/3) * m1 * L1**2  # Rod 1 about its pivot (end)
        I2 = (1/3) * m2 * L2**2  # Rod 2 about its pivot (end)
        
        delta = theta2 - theta1
        
        # Derivatives of angles
        dtheta1 = omega1
        dtheta2 = omega2
        
        # Build the mass matrix M
        # M11 includes: rotational inertia of rod 1 + effect of rod 2's mass at end of rod 1
        M11 = I1 + m2 * L1**2
        
        # M22 is just the rotational inertia of rod 2 about its pivot
        M22 = I2
        
        # M12 = M21 represents coupling between the two rods
        M12 = m2 * L1 * Lc2 * np.cos(delta)
        M21 = M12
        
        # Build the C vector (Coriolis/centrifugal terms)
        # These arise from the velocity-dependent terms in the Lagrangian
        C1 = -m2 * L1 * Lc2 * omega2**2 * np.sin(delta)
        C2 = m2 * L1 * Lc2 * omega1**2 * np.sin(delta)
        
        # Build the G vector (gravitational torques)
        G1 = -(m1 * Lc1 + m2 * L1) * g * np.sin(theta1)
        G2 = -m2 * Lc2 * g * np.sin(theta2)
        
        # The equation of motion is: M * omega_dot = G - C
        # So: omega_dot = M^(-1) * (G - C)
        
        # Calculate determinant of mass matrix
        det_M = M11 * M22 - M12 * M21
        
        if abs(det_M) > 1e-10:  # Check for singularity
            # Solve the 2x2 system using matrix inverse
            # [domega1]   [M11 M12]^(-1)   [G1 - C1]
            # [domega2] = [M21 M22]     *  [G2 - C2]
            
            domega1 = (M22 * (G1 - C1) - M12 * (G2 - C2)) / det_M
            domega2 = (M11 * (G2 - C2) - M21 * (G1 - C1)) / det_M
        else:
            # Fallback for near-singular configurations
            domega1 = (G1 - C1) / M11
            domega2 = (G2 - C2) / M22
    
    return [dtheta1, domega1, dtheta2, domega2]




# Energy calculation functions for verification
def calculate_energy_simple(theta1, omega1, theta2, omega2, L1, L2, m1, m2, g=9.81):
    """Calculate total energy for simple (point mass) pendulum"""
    # Kinetic energy using the exact expression
    KE = 0.5 * m1 * (L1 * omega1)**2
    KE += 0.5 * m2 * ((L1 * omega1 * np.cos(theta1) + L2 * omega2 * np.cos(theta2))**2 +
                      (L1 * omega1 * np.sin(theta1) + L2 * omega2 * np.sin(theta2))**2)
    
    # Potential energy
    PE = -m1 * g * L1 * np.cos(theta1)
    PE += -m2 * g * (L1 * np.cos(theta1) + L2 * np.cos(theta2))
    
    return KE + PE


def calculate_energy_compound(theta1, omega1, theta2, omega2, L1, L2, m1, m2, g=9.81):
    """Calculate total energy for compound (distributed mass) pendulum"""
    Lc1 = L1 / 2
    Lc2 = L2 / 2
    
    # Moments of inertia about center of mass
    I1_cm = (1/12) * m1 * L1**2
    I2_cm = (1/12) * m2 * L2**2
    
    # Position of center of mass of rod 1 (x-axis is perpendicular to gravity; so it's useless)
    yc1 = -Lc1 * np.cos(theta1)
    
    # Velocity of center of mass of rod 1
    vxc1 = Lc1 * omega1 * np.cos(theta1)
    vyc1 = Lc1 * omega1 * np.sin(theta1)
    
    # Position of center of mass of rod 2 (x-axis is perpendicular to gravity; so it's useless)
    yc2 = -L1 * np.cos(theta1) - Lc2 * np.cos(theta2)
    
    # Velocity of center of mass of rod 2
    vxc2 = L1 * omega1 * np.cos(theta1) + Lc2 * omega2 * np.cos(theta2)
    vyc2 = L1 * omega1 * np.sin(theta1) + Lc2 * omega2 * np.sin(theta2)
    
    # Kinetic energy = translational + rotational
    KE = 0.5 * m1 * (vxc1**2 + vyc1**2) + 0.5 * I1_cm * omega1**2
    KE += 0.5 * m2 * (vxc2**2 + vyc2**2) + 0.5 * I2_cm * omega2**2
    
    # Potential energy (using center of mass positions)
    PE = m1 * g * yc1 + m2 * g * yc2
    
    return KE + PE

def test_energy_drift():
    # one chaotic run, energy drift check
    L1=L2=1.0; m1=m2=1.0
    y0 = [np.pi/2, 0.0, np.pi/2+0.17, 0.0]         # chaotic
    tspan = (0, 40)
    sol1 = solve_ivp(double_pendulum_deriv, tspan, y0,
                args=(L1, L2, m1, m2, 9.81, 'simple'),
                atol=1e-9, rtol=1e-6, max_step=0.002)
    sol2 = solve_ivp(double_pendulum_deriv, tspan, y0,
                args=(L1, L2, m1, m2, 9.81, 'compound'),
                atol=1e-9, rtol=1e-6, max_step=0.002)
    E1 = [calculate_energy_simple(*s, L1, L2, m1, m2) for s in sol1.y.T]
    E2 = [calculate_energy_compound(*s, L1, L2, m1, m2) for s in sol2.y.T]
    print(f"Simple max |ΔE/E₀| = {max(abs((np.array(E1)-E1[0])/E1[0])):.2e}")
    print(f"Compound max |ΔE/E₀| = {max(abs((np.array(E2)-E2[0])/E2[0])):.2e}")



def simulate_double_pendulum(params):
    """Simulate the double pendulum system using ODE solver"""
    
    # Time parameters
    t_span = (0, params['sim_time'])
    t_eval = np.linspace(0, params['sim_time'], 1000)
    
    # Initial conditions [theta1, omega1, theta2, omega2]
    y0 = [params['theta1'], params['omega1'], params['theta2'], params['omega2']]
    
    # Solve the ODE
    solution = solve_ivp(
        double_pendulum_deriv, 
        t_span, 
        y0, 
        t_eval=t_eval,
        args=(params['L1'], params['L2'], params['m1'], params['m2'], 9.81, params['pendulum_type']),
        method='RK45',
        rtol=1e-6, 
        atol=1e-9
    )
    
    # Extract the solutions
    theta1 = solution.y[0]
    omega1 = solution.y[1]
    theta2 = solution.y[2]
    omega2 = solution.y[3]
    t = solution.t
    
    # Calculate the cartesian coordinates
    x1 = params['L1'] * np.sin(theta1)
    y1 = -params['L1'] * np.cos(theta1)
    
    x2 = x1 + params['L2'] * np.sin(theta2)
    y2 = y1 - params['L2'] * np.cos(theta2)
    
    return t, x1, y1, x2, y2, theta1, theta2, omega1, omega2

class DoublePendulumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Double Pendulum Simulation")
        self.root.geometry("1500x850")
        self.root.minsize(1200, 700)  # Set minimum window size
        self.root.resizable(True, True)  # Allow resizing in both directions
        
        # Initialize parameters
        self.params = {
            'pendulum_type': 'simple',
            'L1': 1.0, 'L2': 1.0,
            'm1': 1.0, 'm2': 1.0,
            'theta1': 120.0 * np.pi / 180,
            'theta2': 120.0 * np.pi / 180,
            'omega1': 0.0, 'omega2': 0.0,
            'sim_time': 20.0
        }
        
        # Animation control variables
        self.current_frame = 0
        self.playing = False
        self.max_frame = 0
        self.step_timer = None
        self.step_direction = 0
        self.dragging_pendulum = None
        self.trace_colors = ['red']
        self.current_trace_index = 0
        self.all_traces = []
        self.parameters_modified = False
        
        # Create GUI first
        self.create_widgets()
        
        # Then simulate initial data
        self.update_simulation()
        
        # Setup animation after data exists
        self.setup_animation()
        
        # Initialize display
        self.update_display(0)
        
        # Bind Enter key for parameter navigation
        self.setup_parameter_navigation()
        
        # Add proper cleanup when window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right side - parameter panel (pack first to maintain width)
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # Maintain minimum width
        
        # Left side - matplotlib plot (fills remaining space)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08)
        self.update_plot_limits()  # Set dynamic limits based on pendulum lengths
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Double Pendulum Simulation', fontsize=18, fontweight='bold')
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Configure matplotlib for better tkinter integration
        self.canvas.draw()
        
        # Create parameter panel
        self.create_parameter_panel(right_frame)
        
        # Create animation objects
        self.create_animation_objects()
        
    def create_parameter_panel(self, parent):
        """Create the parameter input panel"""
        # Title
        title_label = ttk.Label(parent, text="Parameters", font=('Arial', 18, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Pendulum type selection
        type_frame = ttk.LabelFrame(parent, text="Pendulum Type", padding=10)
        type_frame.configure(style='Title.TLabelframe')
        type_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Configure styles for section titles
        style = ttk.Style()
        style.configure('Title.TLabelframe.Label', font=('Arial', 12, 'bold'))
        style.configure('Radio.TRadiobutton', font=('Arial', 11))
        
        self.pendulum_type_var = tk.StringVar(value='simple')
        ttk.Radiobutton(type_frame, text="Simple (Point Mass)", 
                       variable=self.pendulum_type_var, value='simple',
                       command=self.on_pendulum_type_change, style='Radio.TRadiobutton').pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="Compound (Distributed Mass)", 
                       variable=self.pendulum_type_var, value='compound',
                       command=self.on_pendulum_type_change, style='Radio.TRadiobutton').pack(anchor=tk.W)
        
        # Parameter inputs
        params_frame = ttk.LabelFrame(parent, text="Physical Parameters", padding=10)
        params_frame.configure(style='Title.TLabelframe')
        params_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.param_vars = {}
        self.param_entries = {}
        
        # Parameter definitions with validation
        param_defs = [
            ('L1', 'Length L1 (m):', self.params['L1']),
            ('L2', 'Length L2 (m):', self.params['L2']),
            ('m1', 'Mass m1 (kg):', self.params['m1']),
            ('m2', 'Mass m2 (kg):', self.params['m2']),
            ('theta1_deg', 'θ1 (degrees):', self.params['theta1'] * 180 / np.pi),
            ('theta2_deg', 'θ2 (degrees):', self.params['theta2'] * 180 / np.pi),
            ('omega1', 'ω1 (rad/s):', self.params['omega1']),
            ('omega2', 'ω2 (rad/s):', self.params['omega2']),
            ('sim_time', 'Sim time (s):', self.params['sim_time']),
        ]
        
        for i, (key, label, initial_val) in enumerate(param_defs):
            row_frame = ttk.Frame(params_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=label, width=15, font=('Arial', 12)).pack(side=tk.LEFT)
            
            self.param_vars[key] = tk.StringVar(value=f'{initial_val:.1f}')
            entry = ttk.Entry(row_frame, textvariable=self.param_vars[key], width=10, font=('Arial', 12))
            entry.pack(side=tk.RIGHT)
            self.param_entries[key] = entry
            
            # Bind events
            entry.bind('<KeyRelease>', self.on_parameter_change)
            entry.bind('<FocusIn>', self.on_entry_focus)
            
        # Control buttons
        controls_frame = ttk.LabelFrame(parent, text="Animation Controls", padding=10)
        controls_frame.configure(style='Title.TLabelframe')
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        button_frame1 = ttk.Frame(controls_frame)
        button_frame1.pack(fill=tk.X, pady=2)
        
        # Configure button style
        style.configure('Button.TButton', font=('Arial', 10))
        
        self.play_pause_btn = ttk.Button(button_frame1, text="Play", command=self.play_pause, style='Button.TButton', width=8)
        self.play_pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame1, text="Reset", command=self.reset, style='Button.TButton', width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame1, text="Step <<", command=self.step_back, style='Button.TButton', width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame1, text="Step >>", command=self.step_forward, style='Button.TButton', width=8).pack(side=tk.LEFT, padx=5)
        
        button_frame2 = ttk.Frame(controls_frame)
        button_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame2, text="Clear Traces", command=self.clear_traces, style='Button.TButton').pack(side=tk.LEFT)
        
        # Time slider
        slider_frame = ttk.LabelFrame(parent, text="Timeline", padding=10)
        slider_frame.configure(style='Title.TLabelframe')
        slider_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.time_var = tk.DoubleVar()
        self.time_slider = ttk.Scale(slider_frame, from_=0, to=self.params['sim_time'], 
                                    orient=tk.HORIZONTAL, variable=self.time_var,
                                    command=self.on_slider_change)
        self.time_slider.pack(fill=tk.X)
        
        self.time_label = ttk.Label(slider_frame, text="Time: 0.0s | Frame: 1/1000", font=('Arial', 11))
        self.time_label.pack()
        
    def create_animation_objects(self):
        """Create matplotlib objects for animation"""
        # Rod width for compound pendulum
        self.rod_width = min(self.params['L1'], self.params['L2']) * 0.1
        
        # Create objects for animation (initialize both types)
        self.line, = self.ax.plot([], [], 'o-', lw=2, markersize=8)
        
        # Always create all objects to avoid scope issues
        self.mass1 = plt.Circle((0, 0), radius=0.1*np.sqrt(self.params['m1']), fc='blue', alpha=0.7)
        self.mass2 = plt.Circle((0, 0), radius=0.1*np.sqrt(self.params['m2']), fc='red', alpha=0.7)
        self.rod1 = plt.Polygon([(0,0), (0,0), (0,0), (0,0)], fc='blue', alpha=0.7)
        self.rod2 = plt.Polygon([(0,0), (0,0), (0,0), (0,0)], fc='red', alpha=0.7)
        self.pivot = plt.Circle((0, 0), radius=0.05, fc='black')
        self.joint = plt.Circle((0, 0), radius=0.05, fc='black')
        
        # Always add ALL objects to the axes
        self.ax.add_patch(self.mass1)
        self.ax.add_patch(self.mass2)
        self.ax.add_patch(self.rod1)
        self.ax.add_patch(self.rod2)
        self.ax.add_patch(self.pivot)
        self.ax.add_patch(self.joint)
        
        # Set initial visibility based on pendulum type
        self.update_object_visibility()
        
        # Create initial trace
        self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.5, color='red')
        self.all_traces.append(self.trace)
        
        # Path trace data
        self.trace_x, self.trace_y = [], []
        
        # Add time counter text
        self.time_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=14, fontweight='bold')
        
        # Add energy drift text in bottom right corner
        self.energy_drift_text = self.ax.text(0.95, 0.05, '', transform=self.ax.transAxes, 
                                            fontsize=10, ha='right', va='bottom', alpha=0.8)
        
        # Create highlight circles for pendulum selection (initially invisible)
        self.highlight1 = plt.Circle((0, 0), radius=0.3, fc='none', ec='yellow', linewidth=3, alpha=0.8, visible=False)
        self.highlight2 = plt.Circle((0, 0), radius=0.3, fc='none', ec='yellow', linewidth=3, alpha=0.8, visible=False)
        self.ax.add_patch(self.highlight1)
        self.ax.add_patch(self.highlight2)
        
        # Mouse interaction setup
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        
    def setup_parameter_navigation(self):
        """Setup up/down arrow key navigation between parameters and auto-apply on focus out"""
        param_order = ['L1', 'L2', 'm1', 'm2', 'theta1_deg', 'theta2_deg', 'omega1', 'omega2', 'sim_time']
        
        for i, param in enumerate(param_order):
            entry = self.param_entries[param]
            
            # Bind up/down arrow keys for navigation
            def make_arrow_handler(current_index):
                def on_key(event):
                    if event.keysym == 'Up' and current_index > 0:
                        # Move to previous parameter
                        prev_param = param_order[current_index - 1]
                        self.param_entries[prev_param].focus_set()
                        self.param_entries[prev_param].select_range(0, tk.END)
                    elif event.keysym == 'Down' and current_index < len(param_order) - 1:
                        # Move to next parameter  
                        next_param = param_order[current_index + 1]
                        self.param_entries[next_param].focus_set()
                        self.param_entries[next_param].select_range(0, tk.END)
                    elif event.keysym == 'Return':
                        # Enter key moves to next parameter (keep existing behavior)
                        if current_index < len(param_order) - 1:
                            next_param = param_order[current_index + 1]
                            self.param_entries[next_param].focus_set()
                            self.param_entries[next_param].select_range(0, tk.END)
                        self.on_parameter_submit(param)
                return on_key
            
            # Bind arrow keys and Enter
            entry.bind('<Up>', make_arrow_handler(i))
            entry.bind('<Down>', make_arrow_handler(i))
            entry.bind('<Return>', make_arrow_handler(i))
            
            # Bind focus out event for auto-apply
            def make_focus_out_handler(current_param):
                def on_focus_out(event):
                    self.on_parameter_submit(current_param)
                return on_focus_out
            
            entry.bind('<FocusOut>', make_focus_out_handler(param))
    
    def on_entry_focus(self, event):
        """Handle entry focus - auto-pause and select all text"""
        if self.playing:
            self.playing = False
            self.play_pause_btn.config(text="Play")
            self.update_display(self.current_frame)
            
        # Select all text for easy editing
        event.widget.select_range(0, tk.END)
    
    def on_parameter_change(self, event):
        """Handle parameter changes"""
        self.parameters_modified = True
        
        # Update visual parameters in real-time
        visual_params = ['L1', 'L2', 'theta1_deg', 'theta2_deg']
        param_name = None
        
        for key, entry in self.param_entries.items():
            if entry == event.widget:
                param_name = key
                break
                
        if param_name in visual_params:
            try:
                self.update_live_preview()
            except ValueError:
                pass  # Invalid input, ignore
    
    def on_parameter_submit(self, param_key):
        """Handle parameter submission - auto-apply if parameters were modified"""
        if self.parameters_modified:
            self.apply_parameters()
        else:
            # Update visual display for visual parameters even if not modified
            visual_params = ['L1', 'L2', 'theta1_deg', 'theta2_deg']
            if param_key in visual_params:
                try:
                    self.update_live_preview()
                except ValueError:
                    pass
    
    def update_live_preview(self):
        """Update the pendulum display with current parameter values"""
        try:
            # Get current values from parameter entries
            L1 = float(self.param_vars['L1'].get())
            L2 = float(self.param_vars['L2'].get())
            theta1_deg = float(self.param_vars['theta1_deg'].get())
            theta2_deg = float(self.param_vars['theta2_deg'].get())
            
            # Update plot limits if lengths changed
            if L1 != self.params['L1'] or L2 != self.params['L2']:
                # Calculate and set new limits directly based on current input values
                max_reach = L1 + L2
                limit = max_reach * 1.2
                self.ax.set_xlim(-limit, limit)
                self.ax.set_ylim(-limit, limit)
            
            # Convert to radians and calculate positions
            theta1_rad = theta1_deg * np.pi / 180
            theta2_rad = theta2_deg * np.pi / 180
            
            # Calculate pendulum positions
            pos1_x = L1 * np.sin(theta1_rad)
            pos1_y = -L1 * np.cos(theta1_rad)
            pos2_x = pos1_x + L2 * np.sin(theta2_rad)
            pos2_y = pos1_y - L2 * np.cos(theta2_rad)
            
            # Update display with preview positions
            if self.params['pendulum_type'] == 'simple':
                line_x = [0, pos1_x, pos2_x]
                line_y = [0, pos1_y, pos2_y]
                self.line.set_data(line_x, line_y)
                self.mass1.center = (pos1_x, pos1_y)
                self.mass2.center = (pos2_x, pos2_y)
            else:
                # Update compound pendulum preview
                self.update_compound_display(pos1_x, pos1_y, pos2_x, pos2_y, L1, L2)
            
            self.canvas.draw_idle()
            
        except ValueError:
            pass  # Invalid input, ignore
    
    def update_compound_display(self, pos1_x, pos1_y, pos2_x, pos2_y, L1, L2):
        """Update compound pendulum visual representation"""
        # Calculate rod width
        current_rod_width = min(L1, L2) * 0.1
        half_width = current_rod_width / 2
        
        # Calculate angle for first rod
        theta1_actual = np.arctan2(pos1_y, pos1_x)
        
        # Calculate points for first rod (origin to joint1)
        dx1, dy1 = np.cos(theta1_actual), np.sin(theta1_actual)
        px1, py1 = -dy1, dx1
        
        rod1_points = [
            (0 - half_width * px1, 0 - half_width * py1),
            (0 + half_width * px1, 0 + half_width * py1),
            (pos1_x + half_width * px1, pos1_y + half_width * py1),
            (pos1_x - half_width * px1, pos1_y - half_width * py1)
        ]
        self.rod1.set_xy(rod1_points)
        
        # Calculate angle for second rod
        theta2_actual = np.arctan2(pos2_y - pos1_y, pos2_x - pos1_x)
        
        # Calculate points for second rod (joint1 to joint2)
        dx2, dy2 = np.cos(theta2_actual), np.sin(theta2_actual)
        px2, py2 = -dy2, dx2
        
        rod2_points = [
            (pos1_x - half_width * px2, pos1_y - half_width * py2),
            (pos1_x + half_width * px2, pos1_y + half_width * py2),
            (pos2_x + half_width * px2, pos2_y + half_width * py2),
            (pos2_x - half_width * px2, pos2_y - half_width * py2)
        ]
        self.rod2.set_xy(rod2_points)
        
        # Update pivot and joint positions
        self.pivot.center = (0, 0)
        self.joint.center = (pos1_x, pos1_y)
    
    def on_pendulum_type_change(self):
        """Handle pendulum type change"""
        new_type = self.pendulum_type_var.get()
        if new_type != self.params['pendulum_type']:
            if self.playing:
                self.playing = False
                self.play_pause_btn.config(text="Play")
                self.update_display(self.current_frame)
            
            self.params['pendulum_type'] = new_type
            self.parameters_modified = True
            self.update_object_visibility()
            
            # Immediately apply the changes to create new simulation with new trace
            self.apply_parameters()
    
    def update_object_visibility(self):
        """Update object visibility based on pendulum type"""
        if self.params['pendulum_type'] == 'simple':
            # Show simple objects, hide compound objects
            self.mass1.set_visible(True)
            self.mass2.set_visible(True)
            self.line.set_visible(True)
            self.rod1.set_visible(False)
            self.rod2.set_visible(False)
            self.pivot.set_visible(False)
            self.joint.set_visible(False)
        else:
            # Show compound objects, hide simple objects
            self.mass1.set_visible(False)
            self.mass2.set_visible(False)
            self.line.set_visible(False)
            self.rod1.set_visible(True)
            self.rod2.set_visible(True)
            self.pivot.set_visible(True)
            self.joint.set_visible(True)
    
    def update_plot_limits(self):
        """Update plot limits dynamically based on pendulum lengths"""
        # Maximum reach is when both pendulums are fully extended in same direction
        max_reach = self.params['L1'] + self.params['L2']
        # Add 20% padding for better visualization
        limit = max_reach * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
    
    def update_simulation(self):
        """Update simulation data with current parameters"""
        self.t, self.x1, self.y1, self.x2, self.y2, self.theta1_sim, self.theta2_sim, self.omega1_sim, self.omega2_sim = simulate_double_pendulum(self.params)
        self.max_frame = len(self.t) - 1
        self.current_frame = 0
        
        # Calculate initial energy for drift tracking
        if self.params['pendulum_type'] == 'simple':
            self.initial_energy = calculate_energy_simple(
                self.theta1_sim[0], self.omega1_sim[0], self.theta2_sim[0], self.omega2_sim[0],
                self.params['L1'], self.params['L2'], self.params['m1'], self.params['m2']
            )
        else:
            self.initial_energy = calculate_energy_compound(
                self.theta1_sim[0], self.omega1_sim[0], self.theta2_sim[0], self.omega2_sim[0],
                self.params['L1'], self.params['L2'], self.params['m1'], self.params['m2']
            )
        
        # Update slider range (only if slider exists)
        if hasattr(self, 'time_slider'):
            self.time_slider.config(to=self.params['sim_time'])
        
        # Clear current trace data
        self.trace_x.clear()
        self.trace_y.clear()
        
        # Create new trace with different color
        new_color = self.get_random_trace_color()
        self.trace_colors.append(new_color)
        self.current_trace_index += 1
        
        if len(self.all_traces) > self.current_trace_index:
            # Reuse existing trace
            self.trace = self.all_traces[self.current_trace_index]
            self.trace.set_color(new_color)
        else:
            # Create new trace
            self.trace, = self.ax.plot([], [], '-', lw=1, alpha=0.5, color=new_color)
            self.all_traces.append(self.trace)
    
    def get_random_trace_color(self):
        """Generate a random color for new traces"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        return random.choice([c for c in colors if c not in self.trace_colors[-5:]])
    
    def apply_parameters(self):
        """Apply current parameters and create new simulation"""
        try:
            # Validate and get all parameters
            new_params = {}
            new_params['pendulum_type'] = self.pendulum_type_var.get()
            
            for key in ['L1', 'L2', 'm1', 'm2', 'omega1', 'omega2', 'sim_time']:
                value = float(self.param_vars[key].get())
                if key in ['L1', 'L2', 'm1', 'm2', 'sim_time'] and value <= 0:
                    raise ValueError(f"{key} must be positive")
                new_params[key] = value
            
            new_params['theta1'] = float(self.param_vars['theta1_deg'].get()) * np.pi / 180
            new_params['theta2'] = float(self.param_vars['theta2_deg'].get()) * np.pi / 180
            
            # Update parameters
            self.params.update(new_params)
            
            # Update rod width for compound pendulum
            self.rod_width = min(self.params['L1'], self.params['L2']) * 0.1
            
            # Update plot limits for new pendulum lengths
            self.update_plot_limits()
            
            # Update object visibility
            self.update_object_visibility()
            
            # Generate new simulation
            self.update_simulation()
            
            # Update display
            self.update_display(0)
            
            # Reset modified flag
            self.parameters_modified = False
            
            # Update title
            title_str = f"Double Pendulum Simulation ({self.params['pendulum_type'].capitalize()})\n"
            title_str += f"L1={self.params['L1']}m, L2={self.params['L2']}m, m1={self.params['m1']}kg, m2={self.params['m2']}kg"
            self.ax.set_title(title_str)
            
            print(f"New simulation created with parameters: {self.params}")
            
        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
    
    def update_display(self, frame_idx):
        """Update the pendulum display for a given frame"""
        i = frame_idx
        
        # Update time display
        self.time_text.set_text(f'Time = {self.t[i]:.1f}s | Frame = {i+1}/{len(self.t)}')
        self.time_label.config(text=f"Time: {self.t[i]:.1f}s | Frame: {i+1}/{len(self.t)}")
        
        # Calculate current angles and angular velocities for display
        current_theta1 = np.arctan2(self.x1[i], -self.y1[i]) * 180 / np.pi
        current_theta2 = np.arctan2(self.x2[i] - self.x1[i], -(self.y2[i] - self.y1[i])) * 180 / np.pi
        
        # Calculate angular velocities (approximate from position differences)
        if i > 0:
            dt = self.t[i] - self.t[i-1]
            if dt > 0:
                prev_theta1 = np.arctan2(self.x1[i-1], -self.y1[i-1])
                prev_theta2 = np.arctan2(self.x2[i-1] - self.x1[i-1], -(self.y2[i-1] - self.y1[i-1]))
                current_theta1_rad = np.arctan2(self.x1[i], -self.y1[i])
                current_theta2_rad = np.arctan2(self.x2[i] - self.x1[i], -(self.y2[i] - self.y1[i]))
                
                # Handle angle wrapping
                dtheta1 = current_theta1_rad - prev_theta1
                dtheta2 = current_theta2_rad - prev_theta2
                
                # Unwrap angles
                if dtheta1 > np.pi:
                    dtheta1 -= 2*np.pi
                elif dtheta1 < -np.pi:
                    dtheta1 += 2*np.pi
                    
                if dtheta2 > np.pi:
                    dtheta2 -= 2*np.pi
                elif dtheta2 < -np.pi:
                    dtheta2 += 2*np.pi
                
                current_omega1 = dtheta1 / dt
                current_omega2 = dtheta2 / dt
            else:
                current_omega1 = 0.0
                current_omega2 = 0.0
        else:
            current_omega1 = self.params.get('omega1', 0.0)
            current_omega2 = self.params.get('omega2', 0.0)
        
        # Update parameter displays with current values only when paused
        if not self.playing:
            self.param_vars['theta1_deg'].set(f'{current_theta1:.1f}')
            self.param_vars['theta2_deg'].set(f'{current_theta2:.1f}')
            self.param_vars['omega1'].set(f'{current_omega1:.2f}')
            self.param_vars['omega2'].set(f'{current_omega2:.2f}')
        
        # Update pendulum display
        if self.params['pendulum_type'] == 'simple':
            # For simple pendulum, update the line and masses
            line_x = [0, self.x1[i], self.x2[i]]
            line_y = [0, self.y1[i], self.y2[i]]
            self.line.set_data(line_x, line_y)
            
            # Update mass positions
            self.mass1.center = (self.x1[i], self.y1[i])
            self.mass2.center = (self.x2[i], self.y2[i])
        else:
            # For compound pendulum, update the rod polygons
            self.update_compound_display(self.x1[i], self.y1[i], self.x2[i], self.y2[i], 
                                        self.params['L1'], self.params['L2'])
        
        # Calculate and display energy drift
        if hasattr(self, 'initial_energy'):
            # Calculate current energy using exact angular velocities
            if self.params['pendulum_type'] == 'simple':
                current_energy = calculate_energy_simple(
                    self.theta1_sim[i], self.omega1_sim[i], self.theta2_sim[i], self.omega2_sim[i],
                    self.params['L1'], self.params['L2'], self.params['m1'], self.params['m2']
                )
            else:
                current_energy = calculate_energy_compound(
                    self.theta1_sim[i], self.omega1_sim[i], self.theta2_sim[i], self.omega2_sim[i],
                    self.params['L1'], self.params['L2'], self.params['m1'], self.params['m2']
                )
            
            # Calculate energy drift percentage
            if abs(self.initial_energy) > 1e-10:
                energy_drift_percent = ((current_energy - self.initial_energy) / self.initial_energy) * 100
                self.energy_drift_text.set_text(f'Energy drift: {energy_drift_percent:.3f}%')
            else:
                self.energy_drift_text.set_text('Energy drift: N/A')
        
        # Update slider position
        self.time_var.set(self.t[i])
        
        self.canvas.draw_idle()
    
    def setup_animation(self):
        """Setup the animation"""
        def animate(frame):
            if self.playing:
                i = self.current_frame
                
                # Update trace
                if self.current_frame == 0 and len(self.trace_x) > 0:
                    self.trace_x.clear()
                    self.trace_y.clear()
                
                # Add current point to trace
                if i < len(self.t):
                    self.trace_x.append(self.x2[i])
                    self.trace_y.append(self.y2[i])
                    current_trace = self.all_traces[self.current_trace_index]
                    current_trace.set_data(self.trace_x, self.trace_y)
                
                # Update display
                self.update_display(i)
                
                # Increment frame counter
                self.current_frame += 1
                
                # Handle looping
                if self.current_frame >= len(self.t):
                    self.current_frame = 0
            
            return [self.line, self.time_text, self.mass1, self.mass2, self.rod1, self.rod2, self.pivot, self.joint, self.highlight1, self.highlight2] + self.all_traces
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, animate, frames=10000,
            interval=25, blit=False, repeat=True, cache_frame_data=False
        )
    
    def play_pause(self):
        """Toggle play/pause"""
        self.playing = not self.playing
        if self.playing:
            self.play_pause_btn.config(text="Pause")
            # Remove focus from any entry widget
            self.root.focus_set()
        else:
            self.play_pause_btn.config(text="Play")
            self.update_display(self.current_frame)
    
    def reset(self):
        """Reset animation to beginning"""
        self.current_frame = 0
        self.playing = False
        self.play_pause_btn.config(text="Play")
        self.trace_x.clear()
        self.trace_y.clear()
        self.trace.set_data(self.trace_x, self.trace_y)
        self.time_var.set(0)
        
        # Reset parameter displays to initial values
        self.param_vars['theta1_deg'].set(f'{self.params["theta1"]*180/np.pi:.1f}')
        self.param_vars['theta2_deg'].set(f'{self.params["theta2"]*180/np.pi:.1f}')
        
        self.update_display(0)
    
    def step_back(self):
        """Step backward one frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.playing = False
            self.play_pause_btn.config(text="Play")
            self.update_trace(self.current_frame)
            self.update_display(self.current_frame)
    
    def step_forward(self):
        """Step forward one frame"""
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            self.playing = False
            self.play_pause_btn.config(text="Play")
            self.update_trace(self.current_frame)
            self.update_display(self.current_frame)
    
    def update_trace(self, up_to_frame):
        """Update the trace up to a given frame"""
        self.trace_x.clear()
        self.trace_y.clear()
        for j in range(up_to_frame + 1):
            self.trace_x.append(self.x2[j])
            self.trace_y.append(self.y2[j])
        self.trace.set_data(self.trace_x, self.trace_y)
    
    def clear_traces(self):
        """Clear all traces"""
        self.trace_x.clear()
        self.trace_y.clear()
        
        for trace_obj in self.all_traces:
            trace_obj.set_data([], [])
        
        self.canvas.draw()
    
    def on_slider_change(self, val):
        """Handle slider value changes"""
        if not self.playing:
            target_time = float(val)
            self.current_frame = np.argmin(np.abs(self.t - target_time))
            self.update_trace(self.current_frame)
            self.update_display(self.current_frame)
    
    def on_mouse_press(self, event):
        """Handle mouse press for pendulum dragging"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            # Always stop animation when clicking anywhere on the plot
            self.playing = False
            self.play_pause_btn.config(text="Play")
            # Remove focus from any entry widget
            self.root.focus_set()
            
            mouse_pos = (event.xdata, event.ydata)
            pendulum_clicked = self.which_pendulum_clicked(mouse_pos, self.current_frame)
            
            if pendulum_clicked:
                self.dragging_pendulum = pendulum_clicked
                # Show highlight for selected pendulum
                pos1, pos2 = self.get_pendulum_positions(self.current_frame)
                if pendulum_clicked == 1:
                    self.highlight1.center = pos1
                    self.highlight1.set_visible(True)
                    self.highlight2.set_visible(False)
                else:
                    self.highlight2.center = pos2
                    self.highlight2.set_visible(True)
                    self.highlight1.set_visible(False)
                self.canvas.draw_idle()
            else:
                # Hide all highlights if clicking empty space
                self.highlight1.set_visible(False)
                self.highlight2.set_visible(False)
                self.dragging_pendulum = None
                self.canvas.draw_idle()
    
    def on_mouse_motion(self, event):
        """Handle mouse motion for pendulum dragging"""
        if self.dragging_pendulum and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            mouse_pos = (event.xdata, event.ydata)
            constrained_pos = self.constrain_pendulum_position(mouse_pos, self.dragging_pendulum)
            
            # Update pendulum position in real-time
            if self.dragging_pendulum == 1:
                # When dragging pendulum 1, maintain the angular relationship of pendulum 2
                current_x1, current_y1 = constrained_pos
                
                # Calculate the original angle of pendulum 2 relative to pendulum 1
                pos1_orig, pos2_orig = self.get_pendulum_positions(self.current_frame)
                original_theta2_rel = np.arctan2(pos2_orig[0] - pos1_orig[0], -(pos2_orig[1] - pos1_orig[1]))
                
                # Calculate new position for pendulum 2 maintaining the same relative angle
                new_pos2_x = current_x1 + self.params['L2'] * np.sin(original_theta2_rel)
                new_pos2_y = current_y1 - self.params['L2'] * np.cos(original_theta2_rel)
                
                # Update display based on pendulum type
                if self.params['pendulum_type'] == 'simple':
                    line_x = [0, current_x1, new_pos2_x]
                    line_y = [0, current_y1, new_pos2_y]
                    self.line.set_data(line_x, line_y)
                    self.mass1.center = (current_x1, current_y1)
                    self.mass2.center = (new_pos2_x, new_pos2_y)
                else:
                    # Update compound pendulum during drag
                    self.update_compound_display(current_x1, current_y1, new_pos2_x, new_pos2_y, 
                                               self.params['L1'], self.params['L2'])
                
                # Update highlight position
                self.highlight1.center = (current_x1, current_y1)
            
            elif self.dragging_pendulum == 2:
                # When dragging pendulum 2, pendulum 1 stays fixed
                constrained_pos2 = constrained_pos
                pos1_current, _ = self.get_pendulum_positions(self.current_frame)
                
                # Update display based on pendulum type
                if self.params['pendulum_type'] == 'simple':
                    line_x = [0, pos1_current[0], constrained_pos2[0]]
                    line_y = [0, pos1_current[1], constrained_pos2[1]]
                    self.line.set_data(line_x, line_y)
                    self.mass2.center = constrained_pos2
                else:
                    # Update compound pendulum during drag
                    self.update_compound_display(pos1_current[0], pos1_current[1], constrained_pos2[0], constrained_pos2[1], 
                                               self.params['L1'], self.params['L2'])
                
                # Update highlight position
                self.highlight2.center = constrained_pos2
            
            self.canvas.draw_idle()
    
    def on_mouse_release(self, event):
        """Handle mouse release to finish pendulum dragging"""
        if self.dragging_pendulum:
            # Hide all highlights
            self.highlight1.set_visible(False)
            self.highlight2.set_visible(False)
            
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                mouse_pos = (event.xdata, event.ydata)
                
                # Get final positions and regenerate simulation
                if self.dragging_pendulum == 1:
                    # When dragging pendulum 1, maintain the angular relationship of pendulum 2
                    final_pos1 = self.constrain_pendulum_position(mouse_pos, 1)
                    
                    # Calculate the original angle of pendulum 2 relative to pendulum 1
                    pos1_orig, pos2_orig = self.get_pendulum_positions(self.current_frame)
                    original_theta2_rel = np.arctan2(pos2_orig[0] - pos1_orig[0], -(pos2_orig[1] - pos1_orig[1]))
                    
                    # Calculate final position for pendulum 2 maintaining the same relative angle
                    final_pos2 = (
                        final_pos1[0] + self.params['L2'] * np.sin(original_theta2_rel),
                        final_pos1[1] - self.params['L2'] * np.cos(original_theta2_rel)
                    )
                elif self.dragging_pendulum == 2:
                    # When dragging pendulum 2, pendulum 1 stays fixed
                    final_pos2 = self.constrain_pendulum_position(mouse_pos, 2)
                    final_pos1, _ = self.get_pendulum_positions(self.current_frame)
                
                # Regenerate simulation with new initial conditions
                self.regenerate_simulation_from_positions(final_pos1, final_pos2)
                self.update_display(0)
            
            self.dragging_pendulum = None
            self.canvas.draw_idle()
    
    def get_pendulum_positions(self, frame_idx):
        """Get current pendulum positions"""
        return (self.x1[frame_idx], self.y1[frame_idx]), (self.x2[frame_idx], self.y2[frame_idx])
    
    def which_pendulum_clicked(self, mouse_pos, frame_idx):
        """Determine which pendulum was clicked with improved detection area"""
        pos1, pos2 = self.get_pendulum_positions(frame_idx)
        
        # Larger click radius for easier selection
        click_radius = 0.4  # Increased from 0.2
        
        # For compound pendulum, also check if click is along the rod
        if self.params['pendulum_type'] == 'compound':
            # Check rod 1 (origin to pos1)
            rod1_dist = self.distance_to_line_segment((0, 0), pos1, mouse_pos)
            # Check rod 2 (pos1 to pos2)
            rod2_dist = self.distance_to_line_segment(pos1, pos2, mouse_pos)
            
            # If clicked on rod, consider it a click on the corresponding pendulum
            if rod1_dist < click_radius:
                return 1
            elif rod2_dist < click_radius:
                return 2
        
        # Check distance to pendulum masses/joints
        dist1 = np.sqrt((mouse_pos[0] - pos1[0])**2 + (mouse_pos[1] - pos1[1])**2)
        dist2 = np.sqrt((mouse_pos[0] - pos2[0])**2 + (mouse_pos[1] - pos2[1])**2)
        
        if dist1 < click_radius and (dist1 < dist2 or dist2 >= click_radius):
            return 1
        elif dist2 < click_radius:
            return 2
        return None
    
    def distance_to_line_segment(self, p1, p2, point):
        """Calculate the distance from a point to a line segment"""
        x1, y1 = p1
        x2, y2 = p2
        px, py = point
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # If the line segment has zero length
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Parameter t for the closest point on the line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on line segment
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def constrain_pendulum_position(self, mouse_pos, pendulum_num):
        """Constrain pendulum position to its length constraints"""
        if pendulum_num == 1:
            distance = np.sqrt(mouse_pos[0]**2 + mouse_pos[1]**2)
            if distance > 0:
                scale = self.params['L1'] / distance
                return (mouse_pos[0] * scale, mouse_pos[1] * scale)
            return (0, -self.params['L1'])
        elif pendulum_num == 2:
            pos1, _ = self.get_pendulum_positions(self.current_frame)
            dx = mouse_pos[0] - pos1[0]
            dy = mouse_pos[1] - pos1[1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 0:
                scale = self.params['L2'] / distance
                return (pos1[0] + dx * scale, pos1[1] + dy * scale)
            return (pos1[0], pos1[1] - self.params['L2'])
    
    def regenerate_simulation_from_positions(self, pos1, pos2):
        """Regenerate simulation with new initial positions"""
        # Calculate new initial angles from positions
        new_theta1 = np.arctan2(pos1[0], -pos1[1])
        new_theta2 = np.arctan2(pos2[0] - pos1[0], -(pos2[1] - pos1[1]))
        
        # Update parameters
        self.params['theta1'] = new_theta1
        self.params['theta2'] = new_theta2
        self.params['omega1'] = 0.0
        self.params['omega2'] = 0.0
        
        # Update parameter displays
        self.param_vars['theta1_deg'].set(f'{new_theta1 * 180 / np.pi:.1f}')
        self.param_vars['theta2_deg'].set(f'{new_theta2 * 180 / np.pi:.1f}')
        
        # Generate new simulation
        self.update_simulation()

    def on_closing(self):
        """Handle window closing"""
        try:
            self.playing = False
            if hasattr(self, 'ani'):
                self.ani.event_source.stop()
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)
        except:
            pass  # Ignore cleanup errors
        finally:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = DoublePendulumApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
