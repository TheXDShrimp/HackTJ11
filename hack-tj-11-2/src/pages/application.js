import React, { useState } from 'react';
import Navbar from "../components/Navbar";
import { useRouter } from 'next/router';





const UploadImage = () => {
    const router = useRouter();
    const [file, setFile] = useState(null);
    const [mediaType, setMediaType] = useState('image');
    const [timeVariable, setTimeVariable] = useState(50);
    const [changeVariable, setChangeVariable] = useState(12);

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

    const handleSubmit = () => {
        console.log('submit');
        router.push('./login');

        return (
            <>
            </>
        );
        // router.push('http://127.0.0.1/5000/attack');

        try {
            const fileInput = document.getElementById('file_input');
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            console.log("created form data")

            const numIterations = document.getElementById('num_iterations').value;
            const strength = document.getElementById('strength').value;
            const mediaType = document.getElementById('default-checkbox').checked ? 'audio' : 'image';

            const response = fetch('http://127.0.0.1:5000/attack', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: file.name,
                    epsilon: numIterations,
                    lr: strength
                })
            });
            console.log("sent request")

            const responseData = response.json();
            console.log("got response")

            if (response.ok) {
                console.log('Attack successful:', responseData.message);
                // Handle success logic here, maybe show a success message
                // start downloading the attacked file.
                const downloadLink = document.createElement('a');
                downloadLink.href = responseData.attackedImageUrl;
                downloadLink.download = 'attacked_image.jpg';
                downloadLink.click();
            }

            else {
                console.error('Error during attack:', responseData.error);
                // Handle error logic here, maybe show an error message to the user
            }
        } catch (error) {
            console.error('Error during form submission:', error);
            // Handle unexpected errors here
        }
    };


    const handleClick = () => {
        console.log('submit');
        router.push('/loading_form');
    };


    return (
        <>
            <Navbar loggedIn={true} />
            <div className="grid grid-cols-2">
                <div className="flex flex-col p-24 pt-36 max-w-screen-sm">
                    <div>
                        <div>
                            <h2 className="text-4xl font-bold mb-4">Protect Data</h2>
                            <label className="block mb-2 text-lg font-medium text-gray-900 text-white" htmlFor="file_input">Upload file</label>
                            <input onChange={handleFileChange} className="block w-full text-md text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 text-gray-400 focus:outline-none bg-gray-700 border-gray-600 placeholder-gray-400 text-white" id="file_input" type="file" />
                        </div>
                        <div>
                            <label className="block mb-2 text-lg font-medium text-gray-900 text-white mt-6">Media Type</label>
                            <div className="flex items-center mb-4">
                                <input onChange={handleMediaTypeChange} checked={mediaType === 'image'} id="default-radio-1" type="radio" value="" name="default-radio" className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 focus:ring-blue-600 ring-offset-gray-800 focus:ring-2 bg-gray-700 border-gray-600" />
                                <label htmlFor="default-radio-1" className="ms-2 text-sm font-medium text-gray-900 text-white">Image</label>
                            </div>
                            <div className="flex items-center text-white">
                                <input onChange={handleMediaTypeChange} checked={mediaType === 'audio'} id="default-radio-2" type="radio" value="" name="default-radio" className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 focus:ring-blue-600 ring-offset-gray-800 focus:ring-2 bg-gray-700 border-gray-600" />
                                <label htmlFor="default-radio-2" className="ms-2 text-sm font-medium text-gray-900 text-white">Audio</label>
                            </div>

                            {/* <input id="default-checkbox" type="checkbox" checked={mediaType === 'audio'} onChange={handleMediaTypeChange} className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-blue-600 ring-offset-gray-800 focus:ring-2 bg-gray-700 border-gray-600" />
                                <label htmlFor="default-checkbox" className="ms-2 text-md font-medium text-gray-900 text-gray-300">Audio</label> */}
                        </div>
                        <div className="my-6">
                            <label htmlFor="num_iterations" className="block mb-2 text-lg font-medium text-gray-900 text-white">Number of iterations (1-100)</label>
                            <input type="number" id="num_iterations" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" value={timeVariable} onChange={handleTimeVariableChange} min={1} max={100} required />
                            {/* <Slider className="w-full" min={1} max={100} defaultValue={50} barClassName='bg-blue-500' /> */}
                        </div>
                        <div>
                            <label htmlFor="strength" className="block mb-2 text-lg font-medium text-gray-900 text-white">Strength of attack (1-100)</label>
                            <input type="number" id="strength" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" value={changeVariable} onChange={handleChangeVariableChange} min={1} max={100} required />
                        </div>
                        <div>
                            <a href="/login">
                                <button onClick={handleClick} className="mt-8 text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Submit</button>
                            </a>
                        </div>
                    </div>
                </div>
                <div className="flex flex-col p-24 pt-36 max-w-screen-sm">
                    <h2 className="text-4xl font-bold mb-4">Uploaded Image</h2>
                    {file && <img src={file} alt="Uploaded" style={{ maxWidth: '100%' }} />}
                </div>
            </div>
        </>
    );
};

export default UploadImage;
