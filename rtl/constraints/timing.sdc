# timing.sdc — SDC timing constraints for Cadence Genus/Innovus (ASIC flow)
#
# Technology assumptions:
#   TSMC 28nm HPC+  (common shuttle target)
#   Target: 500 MHz = 2.0 ns period at worst-case SS 0.9V 125C corner
#   The precision controller is ~100 cells — will likely meet 1 GHz+ after P&R
#
# For other process nodes, scale period:
#   TSMC 65nm:   target 300 MHz → period 3.3 ns
#   TSMC 16nm:   target 800 MHz → period 1.25 ns
#   Sky130 (OSS shuttle): target 100 MHz → period 10 ns

# Primary clock
create_clock -name clk -period 2.0 [get_ports clk]

# Clock uncertainty (jitter + skew budget)
set_clock_uncertainty 0.1 [get_clocks clk]

# Transition time on clock
set_clock_transition 0.05 [get_clocks clk]

# Input delays (from upstream pipeline register to this module's input FF)
set_input_delay  -clock clk -max 0.3 [get_ports {s_valid s_data* s_last}]
set_input_delay  -clock clk -min 0.0 [get_ports {s_valid s_data* s_last}]

# Output delays (from this module's output FF to downstream register)
set_output_delay -clock clk -max 0.3 [get_ports {d_valid d_fp16}]
set_output_delay -clock clk -min 0.0 [get_ports {d_valid d_fp16}]

# Reset: driven from a slow control plane, treat as false path
set_false_path -from [get_ports rst_n]

# Drive strength of input ports (model upstream driver)
set_driving_cell -lib_cell BUFX4 -pin Z [get_ports {s_valid s_data* s_last}]

# Load on output ports (model downstream fanout)
set_load 0.05 [get_ports {d_valid d_fp16}]

# Operating conditions (set to match your PDK worst-case library)
# set_operating_conditions -library <lib_name> -condition SS_0P9V_125C
