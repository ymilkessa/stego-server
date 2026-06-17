# Convenience wrappers around docker compose for the stego server.
# The model is downloaded once into the `stego-models` volume and reused.

.PHONY: build up gui down logs shell clean-model clean

# Compose the base file with the GUI overlay for the visualizer targets.
GUI_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.gui.yml

build:            ## Build the server image
	docker compose build

up:               ## Start the server in the background (model persists)
	docker compose up -d

gui:              ## Run with the --add-gui visualizer (foreground; needs an X server)
	$(GUI_COMPOSE) up

down:             ## Stop and remove the container (model is KEPT)
	docker compose down

logs:             ## Follow server logs (watch the model load on first run)
	docker compose logs -f

shell:            ## Open a shell inside the running container
	docker compose exec stego-server bash

clean-model:      ## Wipe ONLY the downloaded model from disk (keeps image)
	docker compose down
	docker volume rm stego-models

clean:            ## Remove container, network, image, AND the model volume
	docker compose down --rmi local
	-docker volume rm stego-models
