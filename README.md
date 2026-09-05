# Customer Segmentation and Market Analysis

Grouping customers by how they actually behave, rather than by the categories a business assumes they fall into.

The idea behind segmentation is straightforward. A business usually treats its customers as one audience, or splits them by something convenient like age bracket or region. But the useful groupings are often behavioural and not obvious in advance: people who buy rarely and spend heavily, people who buy constantly and spend little, people who arrived once and never returned. Those groups want different things, and finding them means letting the data suggest the boundaries instead of drawing them yourself.

This uses clustering to do exactly that. The algorithm gets no labels and no instructions about what the groups should be. It finds the structure that is already in the data, and the interpretation of what each cluster represents comes afterwards, from looking at what its members have in common.

## Running it

```bash
pip install pandas scikit-learn matplotlib
python da1.py
```

## Adapting it to your own data

`da1.py` is written against a particular dataset shape, so the column names and the number of clusters will need changing for yours. The number of clusters especially. There is no correct answer to how many groups exist in a set of customers, only answers that are more or less useful, and picking it well matters more than any other choice in the script.

## Reading the results honestly

Clustering will always give you clusters. Ask it for five groups and it returns five groups, whether or not the data contains five meaningful ones, and they will look convincing on a chart either way. So the test of a segmentation is never that it produced tidy clusters, it is whether the clusters describe something a business could actually act on differently. If two segments would receive the same treatment, they are one segment wearing two names.
