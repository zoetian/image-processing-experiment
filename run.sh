IMAGE=$1
if [ -z "$IMAGE" ]; then
    echo "IMAGE is empty"
    echo usage: $0 image style style_index
    echo e.g: $0 processing/raw_image/Rome.jpg varied 12
    echo ""
    echo "0-9 for monet, 10-30 for varied."
    exit 1
fi

STYLE=$2
if [ -z "$STYLE" ]; then
    echo "STYLE is empty, defaulting to varied"
    STYLE="varied"
fi

STYLE_INDEX=$3
if [ -z "$STYLE_INDEX" ]; then
    echo "STYLE_INDEX is empty, defaulting to 1"
    STYLE_INDEX=1
fi

$(eval "curl -X POST -H X-style-name:$STYLE -H X-style-index:$STYLE_INDEX --data-binary @$IMAGE 'http://localhost:9999/function/artist' > output/""$STYLE""_""$STYLE_INDEX""_""$(basename $IMAGE)" 2>/dev/null)
