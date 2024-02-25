import React, { useState, useEffect } from 'react';
import Navbar from "../components/Navbar";
import { useRouter } from 'next/router';

const LoadingForm = () => {
    const router = useRouter();
    const [file, setFile] = useState(null);
    const [mediaType, setMediaType] = useState('image');
    const [timeVariable, setTimeVariable] = useState(50);
    const [changeVariable, setChangeVariable] = useState(12);
    const [loading, setLoading] = useState(true); // State to control loading screen
    const [time, setTime] = useState(0);

    // Function to simulate loading
    // const simulateLoading = () => {
    //     return new Promise(resolve => {
    //         setTimeout(() => {
    //             resolve();
    //             setTime(time + 10);
    //         }, 10); // 43 seconds
    //     });
    // };

    // Function to handle form submission
    // const handleSubmit = async () => {
    //     setLoading(true); // Show loading screen
    //     try {
    //         // Simulate loading
    //         await simulateLoading();
    //         // Navigate to '/display' after loading
    //         router.push('/display');
    //     } catch (error) {
    //         console.error('Error while loading:', error);
    //     } finally {
    //         setLoading(false); // Hide loading screen
    //     }
    // };
    
    useEffect(() => {
        setTimeout(() => {
            setTime(time + 10);
        }, 10);
        if (time > 43000) {
            router.push('/display');
        }
    }, [time]);

    const handleFileChange = (event) => {
        const uploadedFile = event.target.files[0];
        setFile(URL.createObjectURL(uploadedFile));
    };

    const handleMediaTypeChange = (event) => {
        setMediaType(event.target.checked ? 'audio' : 'image');
    };

    const handleTimeVariableChange = (event) => {
        setTimeVariable(event.target.value);
    };

    const handleChangeVariableChange = (event) => {
        setChangeVariable(event.target.value);
    };

    return (
        <>
            <Navbar loggedIn={true} />
            {loading ? (
                <div className="min-h-screen justify-center items-center mt-64">
                    {/* <div>Loading...</div> // Display loading screen if loading state is true */}
                    <svg xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink" style={{margin:"auto", background:"none", display:"block", shapeRendering:"auto"}} width="200px" height="200px" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid">
                        <circle cx="50" cy="50" r="0" fill="none" stroke="#93dbe9" stroke-width="2">
                            <animate attributeName="r" repeatCount="indefinite" dur="1s" values="0;40" keyTimes="0;1" keySplines="0 0.2 0.8 1" calcMode="spline" begin="0s"></animate>
                            <animate attributeName="opacity" repeatCount="indefinite" dur="1s" values="1;0" keyTimes="0;1" keySplines="0.2 0 0.8 1" calcMode="spline" begin="0s"></animate>
                        </circle>
                        <circle cx="50" cy="50" r="0" fill="none" stroke="#689cc5" stroke-width="2"></circle>
                    </svg>
                </div>
            ) : (
                <>
                    {/* Your form content here */}
                    {/* <button onClick={handleSubmit}>Submit</button> */}
                </>
            )}
        </>
    );
};

export default LoadingForm;
