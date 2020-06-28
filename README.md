## Project overview
This program is used to generate a Bayesian network representing event sequences of the accidents correspoding to **FAR_PART 121** that happened from 1982 to 2006 as reported in the National Transportation Safety Board (NTSB). The SQL query used to pull the accident data is saved in the **SQL_query.txt**. The program developed here builds an end-to-end network to demonstrate the escalation of en event from an incident as an aviation accident. The network starts with the contributory factor and causes of each event, and ends with event outcomes as represented by two separate individual perspectives: aircraft damage and personnel injury. The program generates a file **NTSB.xdsl**, and the file can be loaded into the software [GeNIE modeler](https://www.bayesfusion.com/genie/) for Bayesian data analytics. 

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
pip install -r requirements.txt
```

Besides installing the necessary packages, the software [Graphviz](https://graphviz.org/) needs to be installed for generating graphs.


