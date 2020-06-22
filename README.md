# RNN-Based Node Selection for Sensor Networks with Energy Harvesting, ICTC'18

By Myeung-Un Kim and Hyun Jong Yang.

## Abstract 

A novel recurrent neural network (RNN) based node selection is proposed for sensor networks with energy harvesting, where the downlink (DL) simultaneous wireless information and power transfer (SWIPT) and uplink (UL) wireless powered communication network (WPCN) concepts are jointly considered. While a master node (MN) has a reliable power source, each slave node (SN) is powered by a battery which is charged by energy harvesting. The SN consumes the energy when it senses and transmits data. In addition, all the nodes including the MN have packets to transmit randomly, and every packet generated has its own random deadline. The MN sequentially decides which SN transmits UL data or receives DL data while minimizing the UL transmission failures due to low battery level and DL/UL transmission failures because of exceeded UL/DL packet deadlines. The unpredictability of 1) future channel condition, 2) battery levels, and 3) packet deadlines of SNs makes the node selection problem challenging. In this paper, we propose an RNN-based node selection algorithm in pursuit of minimizing the transmission failures due to low battery level and exceeded UL/DL deadline. Simulation results show that the proposed scheme exhibits lower transmission penalty count than the existing schemes.

[[Paper]](https://ieeexplore.ieee.org/document/8539707)

If you find our work useful in your research, please consider cite:
```bibtex
@INPROCEEDINGS{8539707,
  author={M. U. {Kim} and H. {Jong Yang}},
  booktitle={2018 International Conference on Information and Communication Technology Convergence (ICTC)}, 
  title={RNN-Based Node Selection for Sensor Networks with Energy Harvesting}, 
  year={2018},
  pages={1316-1318},}
```
Feel free to contact Myeung Un Kim (myoungunkim92@gmail.com) if you have any questions!
