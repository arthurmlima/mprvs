-- file: pixel_mem.vhd
-- description: pixel array per processE_ing element
-- pixel E_indexE_ing - x coordE_inate: column
--		  - y coordE_inate: row
--                - E_intermediary image E_index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity E_dec is

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

port(

	clk                         : in std_logic;
	rst                         : in std_logic;
	E_ready_DEC_EB             : inout std_logic := '1'; --sinal indica se o decoder est? pronto para receber a mensagem 
	E_stall_out_DEC            : inout std_logic := '0';
	E_stall_in_DEC             : in std_logic ; 
    E_in_x_dest                : in std_logic_vector(img_width-1 downto 0);
    E_in_y_dest                : in std_logic_vector(img_height-1 downto 0);
    E_in_x_orig                : in std_logic_vector(img_width-1 downto 0);
    E_in_y_orig                : in std_logic_vector(img_height-1 downto 0);
    E_in_fb                    : in std_logic;
    E_out_send                 : out std_logic := '0';
    E_out_direction            : out std_logic_vector(5 downto 0);
    E_valid_EB_DEC             : in      std_logic;
    E_out_new_EB_DEC           : in std_logic
    
);
end entity E_dec;

architecture behavioral of E_dec is

    --signal E_out_direction : std_logic_vector(5 downto 0);--hotspot: PixMem, PE, N, S, E, W
    
    --signal pixel : natural := 0;
    signal x_dest: natural ;
    signal y_dest: natural ;
    signal x_dest_orig: natural ;
    signal y_dest_orig: natural ;
    --signal step  : natural := 0;
    --signal frame : natural := 0;
    signal x_orig: natural;
    signal y_orig: natural;
    --signal fb    : std_logic := '0';
    
    constant x_end : natural := x_init + subimg_width-1; -- end of region
    constant y_end : natural := y_init + subimg_height-1; -- end of region
    
    signal x_diff : integer;-- := x_dest-x_E_init; -- distance to destE_iny
    signal y_diff : integer;-- := y_dest-y_E_init; -- distance to destE_iny
    
    signal NW : std_logic := '0';
    signal SW : std_logic := '0';
    signal NE : std_logic := '0';
    signal SE : std_logic := '0';
    
    signal counter : natural := 0;
    signal state : std_logic := '0';
begin
    
    x_dest <= to_integer(unsigned(E_in_x_dest));
    y_dest <= to_integer(unsigned(E_in_y_dest));
    x_orig <= to_integer(unsigned(E_in_x_orig));
    y_orig <= to_integer(unsigned(E_in_y_orig));
    x_diff <= x_dest-x_init;
    y_diff <= y_dest-y_init;    

process(clk,rst) is
begin
    if(rst='1')then
        E_out_direction <= "000000";

        E_out_send    <= '0';
   
    else
       if(rising_edge(clk))then
       
       if(E_out_new_EB_DEC = '1')then --
       state<= '0';
 --
       end if; 
       
       case state is 
       
       when '0' =>
            if(E_out_new_EB_DEC = '1')then --  
            state<='1';
            E_stall_out_DEC <= '1';
            
                   if(E_in_fb='0')then

                           if( (x_dest>=x_init)and(x_dest<=x_end)and(y_dest>=y_init)and(y_dest<=y_end) ) then
                                
                                    E_out_direction <= "100000"; 

                                   E_out_send    <= '1';
                                   E_ready_DEC_EB <= '0';

                                   
                           elsif( (x_dest>=x_init)and(x_dest<=x_end) )then -- rE_oute E_in Y (North/SE_outh)
                               -- the above condition says that the message is E_in the same X range of the local tile
                               -- defE_ine now if rE_oute North or SE_outh
                               if( y_dest<y_init )then -- rE_oute North
                                   E_out_direction <= "001000";

                                   E_out_send    <= '1';  
                                   E_ready_DEC_EB <= '0';                                    
                               else -- rE_oute SE_outh
                                    E_out_direction <= "000100";    
        
                                    E_out_send    <= '1';   
                                    E_ready_DEC_EB <= '0';                                       
                               end if;
                           elsif( (y_dest>=y_init)and(y_dest<=y_end) )then -- rE_oute E_in X (East/West)
                                -- the above condition says that the message is E_in the same Y range of the local tile
                                -- defE_ine now if rE_oute East or West
                                if( x_dest<x_init)then -- rE_oute West
                                    E_out_direction <= "000001";   
      
                                    E_out_send    <= '1';      
                                    E_ready_DEC_EB <= '0';                                    
                                else -- rE_oute East
                                    E_out_direction <= "000010";       
          
                                    E_out_send    <= '1';          
                                    E_ready_DEC_EB <= '0';                                    
                                end if;
                           elsif( (x_dest<x_init)and(y_dest<y_init) )then -- North-West region
                                --defE_ine now if rE_oute North or West first
                                if(NW='0') then -- rE_oute North
                                    E_out_direction <= "001000";

                                    E_out_send    <= '1';      
                                    E_ready_DEC_EB <= '0';                           
                                else -- rE_oute West
                                    E_out_direction <= "000001";   
      
                                    E_out_send    <= '1';      
                                    E_ready_DEC_EB <= '0';                          
                                end if;
                                NW <= not(NW);
                           elsif( (x_dest<x_init)and(y_dest>y_end) )then -- SE_outh-West region
                                -- defE_ine now if rE_oute SE_outh or West first
                                if(SW='0')then -- rE_oute SE_outh
                                    E_out_direction <= "000100";    
         
                                    E_out_send    <= '1';      
                                    E_ready_DEC_EB <= '0';                                 
                                else -- rE_oute West
                                    E_out_direction <= "000001";   
         
                                    E_ready_DEC_EB <= '0';                       
                                end if;       
                                SW <= not(SW);                    
                           elsif( (x_dest>x_end)and(y_dest<y_init) )then -- North-East region
                                -- defE_ine now if rE_oute North or East first
                                if(NE='0')then -- rE_oute North
                                    E_out_direction <= "001000";

                                    E_out_send    <= '1';                                 
                                    E_ready_DEC_EB <= '0';
                                else -- rE_oute East
                                    E_out_direction <= "000010";       
          
                                    E_out_send    <= '1';    
                                    E_ready_DEC_EB <= '0';                              
                                end if;
                                NE <= not(NE);
                           elsif( (x_dest>x_end)and(y_dest>y_end) )then -- SE_outh-East region
                                -- defE_ine now if rE_oute SE_outh or East first
                                if(SE='0') then -- rE_oute SE_outh
                                    E_out_direction <= "000100";    
        
                                    E_out_send    <= '1';              
                                    E_ready_DEC_EB <= '0';                         
                                else  -- rE_oute East
                                    E_out_direction <= "000010";       
           
                                    E_out_send    <= '1';          
                                    E_ready_DEC_EB <= '0';                        
                                end if;
                                SE <= not(SE);
                           end if;
                           
                       else
                      
                       --------------- FB = 1 = backward message
                        --E_out_send <= '0';

                          if( (x_orig>=x_init)and(x_orig<=x_end)and(y_orig>=y_init)and(y_orig<=y_end) )then
--            
                                   E_out_direction <= "010000"; -- rE_oute to ProcessE_ing Element

                                   E_out_send    <= '1';   
                                   E_ready_DEC_EB <= '0';                         
                           
                                  
                          elsif( (x_orig>=x_init)and(x_orig<=x_end) )then
                              -- the above condition says that the message is E_in the same X range of the local tile
                              -- defE_ine now if rE_oute North or SE_outh
                              if( y_orig<y_init )then -- rE_oute North
                                  E_out_direction <= "001000";

                                  E_out_send    <= '1';     
                                  E_ready_DEC_EB <= '0';                               
                              else -- rE_oute SE_outh
                                   E_out_direction <= "000100";    
         
                                   E_out_send    <= '1';         
                                   E_ready_DEC_EB <= '0';                                
                              end if;
                          elsif( (y_orig>=y_init)and(y_orig<=y_end) )then -- rE_oute E_in X (East/West)
                               -- the above condition says that the message is E_in the same Y range of the local tile
                               -- defE_ine now if rE_oute East or West
                               if( x_orig<x_init)then -- rE_oute West
                                   E_out_direction <= "000001";   
        
                                   E_out_send    <= '1';           
                                   E_ready_DEC_EB <= '0';                               
                               else -- rE_oute East
                                   E_out_direction <= "000010";       
            
                                   E_out_send    <= '1';             
                                   E_ready_DEC_EB <= '0';                                 
                               end if;
                          elsif( (x_orig<x_init)and(y_orig<y_init) )then -- North-West region
                               --defE_ine now if rE_oute North or West first
                               if(NW='0')then -- rE_oute North
                                   E_out_direction <= "001000";

                                   E_out_send    <= '1';         
                                   E_ready_DEC_EB <= '0';                        
                               else -- rE_oute West
                                   E_out_direction <= "000001";   
       
                                   E_out_send    <= '1';     
                                   E_ready_DEC_EB <= '0';                          
                               end if;
                               NW <= not(NW);
                          elsif( (x_orig<x_init)and(y_orig>y_end) )then -- SE_outh-West region
                               -- defE_ine now if rE_oute SE_outh or West first
                               if(SW='0')then -- rE_oute SE_outh
                                   E_out_direction <= "000100";    
         
                                   E_out_send    <= '1';     
                                   E_ready_DEC_EB <= '0';                                 
                               else -- rE_oute West
                                   E_out_direction <= "000001";   
        
                                   E_out_send    <= '1';         
                                   E_ready_DEC_EB <= '0';                       
                               end if;       
                               SW <= not(SW);                    
                          elsif( (x_orig>x_end)and(y_orig<y_init) )then -- North-East region
                               -- defE_ine now if rE_oute North or East first
                               if(NE='0') then -- rE_oute North
                                   E_out_direction <= "001000";

                                   E_out_send    <= '1';                                 
                                   E_ready_DEC_EB <= '0';
                               else -- rE_oute East
                                   E_out_direction <= "000010";       
            
                                   E_out_send    <= '1';        
                                   E_ready_DEC_EB <= '0';                        
                               end if;
                               NE <= not(NE);
                          elsif( (x_orig>x_end)and(y_orig>y_end) )then -- SE_outh-East region
                               -- defE_ine now if rE_oute SE_outh or East first
                               if(SE='0') then -- rE_oute SE_outh
                                   E_out_direction <= "000100";    
        
                                   E_out_send    <= '1';    
                                   E_ready_DEC_EB <= '0';                                
                               else -- rE_oute East
                                   E_out_direction <= "000010";       
            
                                   E_out_send    <= '1';    
                                    E_ready_DEC_EB <= '0';                              
                               end if;
                               SE <= not(SE);
                          end if;
                       
                       
                       end if;-- end fb
                   --end if;
                 --  counter <= 1;
                   
               end if;--E_in ack=0 and E_in_new=1
       when '1' =>

                if(E_stall_in_DEC = '1')then
                
                    E_ready_DEC_EB <= '1'; -- indica para o EB  que o decoder est? livre para receber msg
                    state<='0';
                    E_out_send <= '0';
                    E_stall_out_DEC <= '0';    
                                   
                end if;
                
           when others =>
           
           end case;
        end if;
            
    end if;
end process;


end architecture behavioral;
