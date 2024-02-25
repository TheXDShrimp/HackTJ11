// dbHandler.js

export const handleSubmit = (name, email, password, router) => {
    return fetch('http://127.0.0.1:5000/signup', {
      method: 'POST',
      body: JSON.stringify({
        name: name,
        email: email,
        password: password
      }),
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
    })
    .then((res) => {
      console.log(res.data);
  
      if (!res.ok) {
        throw new Error('Network response was not ok');
      }
  
      return res.json();
    })
    .then(data => {
      console.log(data);
      router.push('/login');
    })
    .catch((err) => {
      console.error('Error signing up:', err)
    });
  };
  