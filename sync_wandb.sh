for dir in wandb/*/; do
    wandb sync "$dir" -e vdaita
done