// // pages/api/auth/google.js

// import { google } from 'googleapis';

// export default async function handler(req, res) {
//   const { code } = req.query;

//   // Exchange authorization code for access token
//   const oauth2Client = new google.auth.OAuth2(
//     process.env.GOOGLE_CLIENT_ID,
//     process.env.GOOGLE_CLIENT_SECRET,
//     process.env.GOOGLE_REDIRECT_URI
//   );

//   try {
//     const { tokens } = await oauth2Client.getToken(code);
//     const accessToken = tokens.access_token;

//     // Use the access token to make requests to Google's API
//     const userInfo = await getUserInfo(accessToken);

//     // Return user information to the client
//     res.status(200).json(userInfo);
//   } catch (error) {
//     console.error('Error exchanging code for token:', error.message);
//     res.status(500).json({ error: 'Failed to exchange code for token' });
//   }
// }

// async function getUserInfo(accessToken) {
//   const oauth2 = google.oauth2({
//     version: 'v2',
//     auth: accessToken,
//   });

//   const { data } = await oauth2.userinfo.get();

//   return {
//     name: data.name,
//     email: data.email,
//     picture: data.picture,
//   };
// }