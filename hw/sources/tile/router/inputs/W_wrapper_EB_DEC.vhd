----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/29/2019 07:35:14 PM
-- Design Name: 
-- Module Name: wrapper_EB_DEC - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;




entity W_wrapper_EB_DEC is
generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);
  Port (
clk                        : in std_logic ;
reset                        : in std_logic ;
--------Handshakes da entrada do EB
--P_W_valid_CF_EB              : in    std_logic; -- indica msg dispon?vel pra enviar ao EB
P_W_ready_EB_CF              : out   std_logic; -- indica que o EB pode receber a mensagem 
P_W_stall_in_EB              : in    std_logic ; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
P_W_stall_out_EB             : inout std_logic;  -- indica que o '''
---------handshakes na sa?da do decoder
P_W_out_send                 : out  std_logic;
--P_W_ready_CF_DEC             : in std_logic;
P_W_stall_out_DEC            : inout std_logic ;
P_W_stall_in_DEC             : in std_logic ;
-------------------------------------------
--P_W_in_ack : in std_logic;

P_W_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0);
P_W_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0);
P_W_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0);
P_W_data_in_step      : in std_logic_vector(n_steps-1 downto 0);
P_W_data_in_frame     : in std_logic_vector(n_frames-1 downto 0);
P_W_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0);
P_W_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0);
P_W_data_in_fb        : in std_logic;
 
P_W_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
P_W_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
P_W_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
P_W_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
P_W_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
P_W_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
P_W_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
P_W_data_out_fb        : out std_logic   ;
--Wrapper_W_ready_EB_CF  : in std_logic;
P_W_out_direction      : out std_logic_vector(5 downto 0);
P_W_ack                : out std_logic;
P_W_req                : in std_logic
   
  
   );
end W_wrapper_EB_DEC;

architecture Behavioral of W_wrapper_EB_DEC is


signal	   w_W_ready_DEC_EB             :  std_logic; -- sinal indica se o decoder est? pronto para receber a mensagem 
signal     w_W_x_dest                   :  std_logic_vector(img_width-1 downto 0);
signal     w_W_y_dest                   :  std_logic_vector(img_height-1 downto 0);
signal     w_W_x_orig                   :  std_logic_vector(img_width-1 downto 0);
signal     w_W_y_orig                   :  std_logic_vector(img_height-1 downto 0);
signal     w_W_fb                       :  std_logic;
signal     w_W_out_send                 :  std_logic;
signal     w_W_ready_CF_DEC             :  std_logic;
signal     w_W_out_direction            :  std_logic_vector(5 downto 0);
SIGnal     w_W_valid_EB_DEC   :  std_logic;
SIGnal     w_W_out_new_EB_DEC  :  std_logic;

signal    s_W_data_pixel     :  std_logic_vector(pix_depth-1 downto 0);  
signal    s_W_data_x_dest    :  std_logic_vector(img_width-1 downto 0);  
signal    s_W_data_y_dest    :  std_logic_vector(img_height-1 downto 0); 
signal    s_W_data_step      :  std_logic_vector(n_steps-1 downto 0);    
signal    s_W_data_frame     :  std_logic_vector(n_frames-1 downto 0);   
signal    s_W_data_x_orig    :  std_logic_vector(img_width-1 downto 0);  
signal    s_W_data_y_orig    :  std_logic_vector(img_height-1 downto 0); 
signal    s_W_data_fb        :  std_logic   ;                                                                    
signal    s_W_ack                :  std_logic;                               
signal    s_W_req                : std_logic;                              





      
begin

Inst_dec : entity work.W_dec

generic map(
                                    x_init          => x_init       ,
                                    y_init          => y_init       ,
                                    img_width       => img_width    ,
                                    img_height      => img_height   ,
                                    n_frames        => n_frames     ,
                                    n_steps         => n_steps      ,
                                    pix_depth       => pix_depth    ,
                                    subimg_width    => subimg_width ,
                                    subimg_height   => subimg_height,
                                    buffer_length   => buffer_length
                                    )

port map (


clk                   =>  clk ,
rst                   => reset ,
W_stall_out_DEC      =>   P_W_stall_out_DEC ,
W_stall_in_DEC       =>   P_W_stall_in_DEC  ,
W_in_x_dest          =>  w_W_x_dest,
W_in_y_dest          =>  w_W_y_dest,
W_in_x_orig          =>  w_W_x_orig,
W_in_y_orig          =>  w_W_y_orig,
W_in_fb              =>  w_W_fb    ,
W_out_direction      =>  P_W_out_direction,
W_out_send           => P_W_out_send   ,
W_ready_DEC_EB       => w_W_ready_DEC_EB,
W_valid_EB_DEC       => w_W_valid_EB_DEC,
W_out_new_EB_DEC     => w_W_out_new_EB_DEC

);

Inst_elastic_buffer : entity work.W_elastic_buffer

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    ,
subimg_width    => subimg_width ,
subimg_height   => subimg_height,
buffer_length   => buffer_length
)
port map (


clk                     =>     clk , 
rst                     =>     reset ,                
W_valid_EB_DEC         =>     w_W_valid_EB_DEC,
W_ready_DEC_EB         =>     w_W_ready_DEC_EB, 
W_out_new_EB_DEC       =>     w_W_out_new_EB_DEC,  
W_ready_EB_CF          =>     P_W_ready_EB_CF,   
W_stall_in_EB          =>     P_W_stall_in_EB,
W_stall_out_EB         =>     P_W_stall_out_EB,                    
W_data_in_pixel        =>     s_W_data_pixel  ,
W_data_in_x_dest       =>     s_W_data_x_dest ,
W_data_in_y_dest       =>     s_W_data_y_dest ,
W_data_in_step         =>     s_W_data_step   ,
W_data_in_frame        =>     s_W_data_frame  ,
W_data_in_x_orig       =>     s_W_data_x_orig ,
W_data_in_y_orig       =>     s_W_data_y_orig ,
W_data_in_fb           =>     s_W_data_fb     ,
W_ack                  =>     s_W_ack         ,
W_req                  =>     s_W_req         ,
W_data_out_pixel       =>     P_W_data_out_pixel ,
W_data_out_x_dest      =>     w_W_x_dest,
W_data_out_y_dest      =>     w_W_y_dest,
W_data_out_step        =>     P_W_data_out_step  ,
W_data_out_frame       =>     P_W_data_out_frame ,
W_data_out_x_orig      =>     w_W_x_orig,
W_data_out_y_orig      =>     w_W_y_orig,
W_data_out_fb          =>     w_W_fb  
--W_ready_CF_EB          =>     Wrapper_W_ready_EB_CF,
);


Inst_in_W_ctrl : entity work.in_W_controller

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    
)
port map (


clk                     =>     clk, 
reset                   =>     reset,       
                                 
oc_W_pixel        =>     P_W_data_in_pixel  ,
oc_W_x_dest       =>     P_W_data_in_x_dest ,
oc_W_y_dest       =>     P_W_data_in_y_dest ,
oc_W_step         =>     P_W_data_in_step   ,
oc_W_frame        =>     P_W_data_in_frame  ,
oc_W_x_orig       =>     P_W_data_in_x_orig,
oc_W_y_orig       =>     P_W_data_in_y_orig,
oc_W_fb           =>     P_W_data_in_fb    ,        
oc_W_new_msg      =>     P_W_req,
oc_W_ack          =>     P_W_ack, 
i_W_pixel         =>     s_W_data_pixel  ,
i_W_x_dest        =>     s_W_data_x_dest ,
i_W_y_dest        =>     s_W_data_y_dest ,
i_W_step          =>     s_W_data_step   ,
i_W_frame         =>     s_W_data_frame  ,
i_W_x_orig        =>     s_W_data_x_orig ,
i_W_y_orig        =>     s_W_data_y_orig ,
i_W_fb            =>     s_W_data_fb     ,
i_W_req           =>     s_W_req         ,
i_W_ack           =>     s_W_ack          

);


P_W_data_out_x_dest <= w_W_x_dest;
P_W_data_out_y_dest <= w_W_y_dest;
P_W_data_out_x_orig <= w_W_x_orig;
P_W_data_out_y_orig <= w_W_y_orig;
P_W_data_out_fb     <= w_W_fb    ;
     
 

   
end Behavioral;
