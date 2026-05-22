#!/usr/bin/env bash
push ../../
git clone git@github.com:ML4PhA-G11/simulators.git
git clone git@github.com:ML4PhA-G11/model-experiments.git
git clone git@github.com:ML4PhA-G11/Apply-NN-KarmanVortexStreet.git
ln -s ../data ./
popd

