To setup the environment, clone the FIDESlib repository locally. And apply the necessary patches as specified inside the `patch/` directory.

Then, to build the docker image navigate to `.devcontainer` and run:

```
docker build -t fideslib .
```

To launch and attach to the container run (you will have to change the path inside the script to match the path where the library is located):

```
./run_container.sh
```

