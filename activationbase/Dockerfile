FROM busybox:stable
# Add the CSV to the image at /activationBase/currentActivation.csv
COPY currentActivation.csv /activationBase/currentActivation.csv

# Default command: list the activationBase folder and print file then sleep (keeps container alive)
CMD ["sh", "-c", "ls -la /activationBase && echo '--- /activationBase/currentActivation.csv ---' && cat /activationBase/currentActivation.csv && sleep 3600"]
