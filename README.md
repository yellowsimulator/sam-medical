# SAM-MEDICAL (Segment-Anything-Medical)

SAM-MEDICAL is a wrapper package for applying SAM (Segment-Anything) on medical images

## Adding segment_anything from Meta as submodule

```python
 git submodule add https://github.com/facebookresearch/segment-anything.git segment_anything
```

## Update the submodule to the latest commit in the upstream repository

```python-repl
git submodule update --remote segment_anything
git add segment_anything
git commit -m "Update submodule segment_anything"

```

## If you clone this repository

```python
git clone --recursive-submodules < this-repo>
```

Using this package:


1-
