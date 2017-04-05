# jones-calculus
Jones calculus for the polarization of light.

Polarized light is represented by a _Jones vector_, and linear optical elements are represented by _Jones matrices_. The state behind an optical element is given by the Jones vector multiplied by the corresponding Jones matrix. Note that the Jones calculus only works for fully polarized light.

See [Wikipedia](https://en.wikipedia.org/wiki/Jones_calculus) for a detailed description.

# Implemented components
* Linear polarizer
* Half-wave plate
* Quarter-wave plate

# Example
Transmit H-polarized light through a half-wave plate at 45 deg: 
```python
import math
from jonescalculus import jonescalculus as jones

jv1 = jones.JonesVector(preset='H')
hwp = jones.HalfWavePlate(math.radians(45))  # Half-wave plate at angle 45 deg wrt x-axis
hwp*jv1
```

