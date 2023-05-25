# Sketch to Render to Market (sketch2render2market)


```bash
docker image build -t sketch2render2market .

docker container run --rm -it --name sketch -v $(pwd)/checkpoints:/sketch2render2market/checkpoints -v $(pwd)/results:/sketch2render2market/results sketch2render2market

docker container exec -it sketch /bin/bash

```

