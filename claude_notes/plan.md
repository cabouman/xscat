# Plan for xscat

Goal: a package that estimates and removes scatter in X-ray CT
measurements, built on mbirtorch in the same design as xcal.

## 1. Repository skeleton — DONE

Packaging, documentation, tests, and developer scripts copied from
the xcal design.

## 2. Review Jingsong's PyTorch code

Jingsong has working PyTorch scatter-correction code.  Read it
before designing anything: what it takes in, what it computes, and
what it returns.

## 3. Concept of operations

Write conops.md: what the user provides, what xscat does, and what
comes back.  Charlie marks it up; iterate until it holds still.

## 4. Design the user API

Write the demo script a user would run, twenty to forty lines,
before implementing anything.  Iterate on that script with Charlie.

## 5. Interface skeleton

Real docstrings, stub bodies, and rendered documentation pages
reviewed as a new user would.

## 6. Implement

Port Jingsong's code behind the agreed interface, verifying against
known ground truth at every stage.
