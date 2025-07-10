| CMPNN                            | GROVER-style CMPNN                               |
|----------------------------------|--------------------------------------------------|
| SMILES → Graph(V, E)            | SMILES → Graph(V, E)                             |
| ↓                               | ↓                                                |
| Init features (atom/bond)       | Init features (atom/bond)                        |
| ↓                               | ↓                                                |
| Message passing (fixed T)       | Dynamic MPN (dyMPN): randomized hops             |
| ↓                               | ↓                                                |
| Aggregator (e.g. sum)           | GTransformer: self-attention over atoms          |
| ↓                               | ↓                                                |
| Readout + FFN                   | Readout + FFN                                    |
| ↓                               | ↓                                                |
| Regression                      | Pretrain (context + motif) → Fine-tune (regression) |


