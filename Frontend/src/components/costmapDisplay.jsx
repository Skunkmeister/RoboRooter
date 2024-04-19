import React from 'react';

const CostmapDisplay = ({ costmapData }) => {
    return (
        <div>
            {costmapData && (
                <div>
                    <h3>Costmap Data:</h3>
                    {/* Assuming costmapData is a base64 encoded image */}
                    <img src={`data:image/png;base64,${costmapData}`} alt="Costmap Image" />
                </div>
            )}
        </div>
    );
}

export default CostmapDisplay;