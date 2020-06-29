[![Build Status](https://travis-ci.com/travis-ci/travis-web.svg?branch=master)](https://travis-ci.com/travis-ci/travis-web)

## Project overview
This program is used to generate a Bayesian network representing event sequences of the accidents correspoding to **FAR_PART 121** that happened from 1982 to 2006 as reported in the National Transportation Safety Board (NTSB). The SQL query used to pull the accident data is saved in the **SQL_query.txt**. The program developed here builds an end-to-end network to demonstrate an event escalating from an incident as an aviation accident. The network starts with the contributory factor and causes of each event, and ends with event outcomes as represented by two separate individual perspectives: aircraft damage and personnel injury. The program generates a file **NTSB.xdsl**, and the file can be loaded into the software [GeNIE modeler](https://www.bayesfusion.com/genie/) for Bayesian data analytics. 

## Requirements
* Python >= 3.6 (Anaconda3 is recommended)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

Besides installing the necessary Python packages, the software [Graphviz](https://graphviz.org/) needs to be installed for generating graphs. If you'd like to disable the function for generating graphs with Graphviz, you can comment the following code at line 731 of the script main.py.

`drawImage(g1)`

## Reference
```
@article{zhang2020bn,
  title={Bayesian network modeling of accident investigation reports for aviation safety assessment},
  author={Zhang, Xiaoge and Mahadevan, Sankaran},
  journal={Reliability Engineering & Systems Safety},
  volume={Under Review},
  year={2020},
  publisher={Elsevier}
}
```



