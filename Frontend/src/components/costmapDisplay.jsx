import React, { useEffect, useState } from 'react';

const CostmapDisplay = ({ costmapData }) => {
    const [normalizedCostmap, setNormalizedCostmap] = useState([]);

    const normalizeCostmap = (costmap) => {
        const minValue = Math.min(...costmap);
        const maxValue = Math.max(...costmap);
        return costmap.map(value => Math.floor((value - minValue) * (255 / (maxValue - minValue))));
    };

    useEffect(() => {
        if (costmapData && costmapData.data) {
            const newNormalizedCostmap = normalizeCostmap(costmapData.data);
            setNormalizedCostmap(newNormalizedCostmap);
        }
    }, [costmapData]);

    const costmapToImageDataUrl = (costmap) => {
        const imageData = new Uint8Array(costmap);
        const blob = new Blob([imageData], { type: 'image/png' });
        return URL.createObjectURL(blob);
    };

    const imageDataUrl = costmapToImageDataUrl(normalizedCostmap);

    return (
        <div>
            <h3>Costmap Display</h3>
            {costmapData && costmapData.data ? (
                <img src={imageDataUrl} alt="Costmap" />
            ) : (
                <p>No costmap data available</p>
            )}
        </div>
    );
}

export default CostmapDisplay;
