# savar
Version 0.4

SAVAR is spatiotemporal stochastic climate model for benchmarking causal discovery methods. 

# Installation
You can install savar using `pip`:

`pip install git+https://github.com/xtibau/savar.git#egg=savar`

# Contributions
Any contribution is more than welcome. If you want to collaborate, do not hesitate to contact me.  Improvements can be made by adding some tutorials with cool data or any other cool idea that you may have. 

# Code Example

```python
from savar.model_generator import SavarGenerator
from copy import deepcopy
from savar.dim_methods import get_varimax_loadings_standard as varimax
import matplotlib.pyplot as plt

resolution = (30, 90)  # Total resolution
N = 3  # total number of modes
savar_generator = SavarGenerator(n_variables=100,
                      n_cross_links=10,
                      time_length=200)
# You need to generate the model
savar_model = savar_generator.generate_savar()
# You need to generate the data
savar_model.generate_data()

# You can use the varimax functions that come with SAVAR
# Or use the package varimax^+ [install it `pip install git+https://github.com/xtibau/varimax_plus.git#egg=varimax_plus`]
modes = varimax(savar_model.data_field.transpose())  # Use variamx to try to recover the weights
for i in range(5):  # Only three are meaningful
    plt.imshow(modes['weights'][:, i].reshape(30, 90))
    plt.colorbar()
    plt.show()
```

# License
SAVAR is a Free Software project under the GNU General Public License v3, which means all its code is available for everyone to download, examine, use, modify, and distribute, subject to the usual restrictions attached to any GPL software. If you are not familiar with the GPL, see the license.txt file for more details on license terms and other legal issues. 

# Cite

Tibau, X., Reimers, C., Gerhardus, A., Denzler, J., Eyring, V., & Runge, J. (2022). A spatiotemporal stochastic climate model for benchmarking causal discovery methods for teleconnections. Environmental Data Science, 1, E12. doi:10.1017/eds.2022.11
