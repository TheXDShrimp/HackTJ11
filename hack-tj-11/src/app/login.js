// pages/login.tsx

import { useState } from 'react';
import { useRouter } from 'next/router';

function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');

  // Function to handle login with Google OAuth
  const handleLoginWithGoogle = () => {
    // Redirect the user to Google OAuth login page
    window.location.href = '/api/auth/google'; // Handle this route in your API routes
  };

  // Function to handle username generation from name
  const generateUsername = (name: string): string => {
    // Logic to convert name to username (e.g., lowercase and remove spaces)
    return name.toLowerCase().replace(/\s+/g, '_');
  };

  // Function to generate a random password
  const generatePassword = (): string => {
    // Logic to generate a random password (e.g., using Math.random())
    return Math.random().toString(36).slice(-8);
  };

  // Function to handle form submission (not shown here)
  const handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    // Logic to handle form submission, possibly sending username and password to server
  };

  return (
    <div>
      <h1>Login Page</h1>
      <button onClick={handleLoginWithGoogle}>Login with Google</button>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default LoginPage;
