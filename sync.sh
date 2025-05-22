

rsync -azi \
    --exclude='.git' \
    --exclude='.idea' \
    --delete \
    . kry45:/data/ebay/data/olivyatan/Workspace/generative_signals


