#!/bin/bash
protoc watch_nonest.proto --python_out=.
protoc watch_nest.proto --python_out=.
protoc watch_nest2.proto --python_out=.
