# Single Motor Example (LEGO Education Python API)
# pip install legoeducation
import legoeducation as le

# Update these to match the Connection Card on your Single Motor
card_color = le.LEGO_COLOR_AZURE
card_serial = '1096'

# Connect to the Single Motor
singlemotor = le.SingleMotor()
singlemotor.connect(card_color=card_color, card_serial=card_serial)

if not singlemotor.connected:
	print('Error connecting to Single Motor.')
	exit(1)

# Run the motor 180 degrees clockwise at 50% speed
singlemotor.motor_run_for_degrees(180, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=50)

singlemotor.disconnect()
exit(0)
