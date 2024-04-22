import React from 'react';
import placeholderImg from './assets/img/placeholder.png'

const DepthDisplay = ({ depthData }) => {

    return (
        <div>
            <div>
                <h3>Depth Camera</h3>
                {depthData ? (
                    <img src={"data:image/jpg;base64," + depthData.data} alt="Depth Image"  />
                ) : (
                   <img src={placeholderImg} alt="No data" />
                )}
            </div>
        </div>
    );
};

export default DepthDisplay;
