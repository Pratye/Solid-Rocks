from solid import *
from solid.utils import *

# ==============================================================
# Board Parameters (in mm)
# ==============================================================
board_length = 50
board_width = 35
board_thickness = 1.6

# Create the PCB as a base plate (colored green)
pcb = color("green")(cube([board_length, board_width, board_thickness]))

# Add mounting holes at the four corners (radius 2 mm, margin 2 mm)
mounting_hole_radius = 1
hole = cylinder(r=mounting_hole_radius, h=board_thickness + 1)
hole1 = translate([2, 2, -0.2])(hole)
hole2 = translate([board_length - 2, 2, -0.2])(hole)
hole3 = translate([board_length - 2, board_width - 2, -0.2])(hole)
hole4 = translate([2, board_width - 2, -0.2])(hole)
pcb = pcb - hole1 - hole2 - hole3 - hole4

# Add Company Label on PCB: "Solid Stones"
# Placed in the center of the board, using white text with a small font.
company_label = translate([board_length/2-2, board_width/2 - 7, board_thickness + 0.3])(
    color("black")(linear_extrude(height=0.3)(text("Solid Stones", size=2.5, halign="center")))
)

# ==============================================================
# Component Definitions (dimensions in mm)
# ==============================================================

# 1. Microchip (Generic) - Black block
microchip_dim = [8, 6, 1.5]
microchip = color("black")(cube(microchip_dim))
# Place near left-center
microchip = translate([5, 16, board_thickness])(microchip)
microchip_label = translate([5, 16, board_thickness + microchip_dim[2]])(
    color("white")(linear_extrude(height=0.3)(text("Chip", size=2, halign="center")))
)

# 2. WiFi Module - Black block
wifi_dim = [12, 8, 1.5]
wifi_module = color("black")(cube(wifi_dim))
# Place in top-left, margin 3 mm from left and adjusted from top mounting hole
wifi_module = translate([3, board_width - wifi_dim[1] - 4, board_thickness])(wifi_module)
wifi_label = translate([3, board_width - wifi_dim[1] - 4, board_thickness + wifi_dim[2]])(
    color("white")(linear_extrude(height=0.3)(text("WiFi", size=2, halign="center")))
)

# 3. Coin Cell Battery - Black cylinder (CR2032-like)
battery_dia = 10
battery_thickness = 2
coin_cell = color("grey")(cylinder(r=battery_dia / 2, h=battery_thickness))
# Place in top-right, margin 3 mm from right edge and adjusted vertically
coin_cell = translate([board_length - battery_dia - 3, board_width - battery_dia - 4, board_thickness])(coin_cell)
battery_label = translate([board_length - battery_dia - 3, board_width - battery_dia - 4, board_thickness + battery_thickness])(
    color("white")(linear_extrude(height=0.3)(text("Batt", size=2, halign="center")))
)

# 4. Wheatstone Bridge for Strain Gauge
# Represented as a 2x2 grid of 3x3x1 mm blocks; bottom right block is the strain gauge
resistor_dim = [3, 3, 1]
resistor = cube(resistor_dim)
bridge_group = union()(
    translate([0, 3, 0])(resistor),  # top left
    translate([3, 3, 0])(resistor),  # top right
    translate([0, 0, 0])(resistor)     # bottom left
)
strain_gauge_block = cube(resistor_dim)  # strain gauge block remains black
bridge_group = union()(bridge_group, translate([3, 0, 0])(strain_gauge_block))
# Place the bridge in bottom-left with a margin of 3 mm
bridge_origin = [3, 3, board_thickness]
bridge = translate(bridge_origin)(bridge_group)
bridge_label = translate([3, 3, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("Bridge", size=2, halign="left")))
)

# 5. Instrumentation Amplifier (AD620) - Black block
amp_dim = [7, 4, 1]
amp = color("black")(cube(amp_dim))
# Place to the right of the bridge with a 2 mm gap
amp = translate([bridge_origin[0] + 6 + 2, bridge_origin[1], board_thickness])(amp)
amp_label = translate([bridge_origin[0] + 6 + 2, bridge_origin[1], board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("Amp", size=2, halign="left")))
)

# 6. Temperature Sensor (LM35) - Black block
temp_dim = [6, 3, 1]
temp_sensor = color("black")(cube(temp_dim))
# Place in bottom-center
temp_sensor = translate([board_length/2 - temp_dim[0]/2, 3, board_thickness])(temp_sensor)
temp_label = translate([board_length/2 - temp_dim[0]/2, 3, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("Temp", size=2, halign="center")))
)

# 7. Vibration Sensor (ADXL335/MPU6050) - Black block
vib_dim = [6, 6, 1]
vib_sensor = color("black")(cube(vib_dim))
# Place in bottom-right
vib_sensor = translate([board_length - vib_dim[0] - 3, 3, board_thickness])(vib_sensor)
vib_label = translate([board_length - vib_dim[0] - 3, 3, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("Vib", size=2, halign="center")))
)

# 8. Humidity Sensor (DHT22) - Black block
hum_dim = [5, 5, 1]
humidity_sensor = color("black")(cube(hum_dim))
# Place above the vibration sensor with a 2 mm gap
humidity_sensor = translate([board_length - hum_dim[0] - 3, 3 + vib_dim[1] + 2, board_thickness])(humidity_sensor)
hum_label = translate([board_length - hum_dim[0] - 3, 3 + vib_dim[1] + 2, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("Hum", size=2, halign="center")))
)

# 9. Power Supply Adjuster - Black block
psu_dim = [8, 4, 1]
psu = color("black")(cube(psu_dim))
# Place to the right of the microchip
psu = translate([5 + microchip_dim[0] + 2, 16, board_thickness])(psu)
psu_label = translate([5 + microchip_dim[0] + 2, 16, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("PSU", size=2, halign="left")))
)

# 10. Diode 1 - Small black block
diode_dim = [3, 1.5, 1]
diode1 = color("black")(cube(diode_dim))
# Place near PSU, above it
diode1 = translate([5 + microchip_dim[0] + 2, 16 + psu_dim[1] + 1, board_thickness])(diode1)
diode1_label = translate([5 + microchip_dim[0] + 2, 16 + psu_dim[1] + 1, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("D1", size=2, halign="left")))
)

# 11. Diode 2 - Small black block
diode2 = color("black")(cube(diode_dim))
# Place near WiFi module, away from the board edge
diode2 = translate([3 + wifi_dim[0] + 1, board_width - wifi_dim[1] - 5, board_thickness])(diode2)
diode2_label = translate([3 + wifi_dim[0] + 1, board_width - wifi_dim[1] - 5, board_thickness + 1])(
    color("white")(linear_extrude(height=0.3)(text("D2", size=2, halign="left")))
)

# 12. Copper Traces (thin black rectangles) connecting components
trace1 = color("brown")(cube([microchip_dim[0] + 2, 0.5, 0.2]))
trace1 = translate([5, 16 + microchip_dim[1]/2, board_thickness + 0.2])(trace1)
trace2 = color("brown")(cube([0.5, 5, 0.2]))
trace2 = translate([5 + microchip_dim[0], board_width - wifi_dim[1] - 5, board_thickness + 0.2])(trace2)
traces = trace1 + trace2

# ==============================================================
# Assemble the Final Sensor Module PCB Model
# ==============================================================

sensor_module = union()(
    pcb,
    company_label,  # Board label with company name "Solid Stones"
    microchip, microchip_label,
    wifi_module, wifi_label,
    coin_cell, battery_label,
    bridge, bridge_label,
    amp, amp_label,
    temp_sensor, temp_label,
    vib_sensor, vib_label,
    humidity_sensor, hum_label,
    psu, psu_label,
    diode1, diode1_label,
    diode2, diode2_label,
    traces
)

# ==============================================================
# Export the Model to a SCAD File
# ==============================================================

scad_render_to_file(sensor_module, 'sensor_module_pcb.scad', file_header='$fn = 50;')
