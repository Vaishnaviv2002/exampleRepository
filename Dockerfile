# Minimal image for activation base
FROM busybox:latest

# Create the activation directory
RUN mkdir -p /activationBase

# Copy the CSV into the image
COPY currentActivation.csv /activationBase/currentActivation.csv

# Keep the container alive (so we can exec in and inspect)
CMD ["sh", "-c", "ls -la /activationBase && echo '--- file contents ---' && cat /activationBase/currentActivation.csv && sleep 3600"]
